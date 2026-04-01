# -*- coding: utf-8 -*-
# Fetch RSS news, score per-sentence sentiment toward TW stocks with lightweight heuristics,
# and persist ALL news (even if no company matched) for downstream analysis / Pages.
import re

import json

import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import feedparser
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# ML-based sentiment analysis
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: transformers/torch not available, using keyword-based sentiment")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONF_DIR = ROOT / "config"
DOCS_DIR = ROOT / "docs"
DOCS_DATA = DOCS_DIR / "data"
NEWS_DIR = DOCS_DATA / "news"  # New directory for daily news
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DATA.mkdir(parents=True, exist_ok=True)
NEWS_DIR.mkdir(parents=True, exist_ok=True)

NEWS_LOG = DATA_DIR / "news_log.jsonl"
# NEWS_JSON = DOCS_DATA / "news_compact.json" # Deprecated
NEWS_INDEX = DOCS_DATA / "news_index.json" # New index file
DAILY_JSON = DOCS_DATA / "daily_sentiment.json"
TODAY_JSON = DOCS_DATA / "today.json"

# ------------------------------
# Config & constants
# ------------------------------
DEFAULT_FEEDS = [
    # Google News - 台灣總覽（你最初提供的主題 RSS）
    "https://news.google.com/rss/topics/CAAqKggKIiRDQkFTRlFvSUwyMHZNRGx6TVdZU0JYcG9MVlJYR2dKVVZ5Z0FQAQ?hl=zh-TW&gl=TW&ceid=TW%3Azh-Hant"
]
MAX_PER_ARTICLE_ABS = 6.0  # 每篇文章對單一股票的分數截斷上限（避免極端值）
WINDOW_CHARS = 12          # 情緒詞與公司別名的匹配視窗（字元）
SENT_SPLIT = r"[。！？!?；;：:\n]"  # 句號/驚嘆/問號/分號/冒號/換行

# 否定詞、轉折詞、條件詞
NEGATION_HINTS = ["不", "未", "無", "非", "否", "別", "毋", "勿", "不是", "沒有", "難以", "難以", "未能"]
CONCESSIVE_LEFT = ["雖", "雖然", "儘管", "即便", "縱使", "就算", "哪怕"]
CONCESSIVE_RIGHT = ["但", "但是", "然而", "卻", "仍", "仍然", "還是", "依然", "不過"]
CONDITIONAL_LEFT = ["除非"]
CONDITIONAL_RIGHT = ["否則", "不然"]

# 情緒詞（可用 config/sentiment_lexicon.json 覆蓋）
DEFAULT_POS = ["成長", "看好", "擴產", "利多", "創新高", "增持", "上修", "優於預期", "超預期", "獲利", "授權", "合作", "擴單", "中標", "得標", "突破", "上調", "加碼", "拓展", "獲選"]
DEFAULT_NEG = ["利空", "衰退", "裁員", "砍單", "下修", "降評", "虧損", "違約", "延遲", "停工", "罰款", "減產", "砍價", "減碼", "疲弱", "下調", "衝擊", "風險"]

# ML model global cache
_sentiment_model = None
_sentiment_tokenizer = None

# ------------------------------
# Utilities
# ------------------------------
def load_json_if_exists(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def normalize_for_match(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("臺", "台")  # 合併「臺/台」
    return s

def split_sentences(text: str, pattern=SENT_SPLIT):
    parts = [p.strip() for p in re.split(pattern, text) if p and p.strip()]
    return parts

def make_news_id(title: str, link: str) -> str:
    raw = normalize_for_match((title or "") + "|" + (link or ""))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def cap_scores(d: dict, cap_abs=MAX_PER_ARTICLE_ABS) -> dict:
    return {k: max(-cap_abs, min(cap_abs, float(v))) for k, v in d.items()}

def load_source_weights() -> dict:
    # 來源可信度權重（可為 {host: number} 或 {host: {pos,neg}}）
    for p in [CONF_DIR / "source_weights.json", DOCS_DATA / "source_weights.json"]:
        obj = load_json_if_exists(p, None)
        if isinstance(obj, dict):
            return obj
    return {}

def weight_for_host(sw: dict, host: str, val: float) -> float:
    if not host:
        return 1.0
    w = sw.get(host)
    if w is None:
        return 1.0
    if isinstance(w, (int, float)):
        return float(w)
    # 支援正/負分離權重
    if isinstance(w, dict):
        return float(w.get("pos", 1.0)) if val >= 0 else float(w.get("neg", 1.0))
    return 1.0

# ------------------------------
# Load company alias / sentiment lexicon
# ------------------------------
def load_company_alias():
    """
    讀取公司別名 → 代碼 的映射：
      1) config/company_alias.json 或 data/company_alias.json (dict format)
      2) data/companies.json (array format: [{code, name, aliases}])
      3) data/companies.csv（欄位：code,name,aliases 逗號/空白分隔）
    皆不存在時回傳空。
    """
    alias_to_codes = defaultdict(set)

    # Try dict format company_alias.json
    for p in [CONF_DIR / "company_alias.json", DATA_DIR / "company_alias.json"]:
        obj = load_json_if_exists(p, None)
        if isinstance(obj, dict):
            for alias, codes in obj.items():
                alias_n = normalize_for_match(alias)
                for c in (codes if isinstance(codes, list) else [codes]):
                    alias_to_codes[alias_n].add(str(c))
            return alias_to_codes

    # Try array format companies.json
    companies_json = DATA_DIR / "companies.json"
    if companies_json.exists():
        try:
            obj = load_json_if_exists(companies_json, None)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        code = str(item.get("code", "")).strip()
                        name = str(item.get("name", "")).strip()
                        aliases = item.get("aliases", [])
                        
                        if code:
                            # Add company name as an alias
                            if name:
                                alias_to_codes[normalize_for_match(name)].add(code)
                            
                            # Add all aliases
                            if isinstance(aliases, list):
                                for alias in aliases:
                                    if alias:
                                        alias_to_codes[normalize_for_match(str(alias))].add(code)
                return alias_to_codes
        except Exception as e:
            print(f"Error loading companies.json: {e}")

    # Try CSV format
    csv_p = DATA_DIR / "companies.csv"
    if csv_p.exists():
        try:
            import csv
            with csv_p.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = str(row.get("code", "")).strip()
                    name = str(row.get("name", "")).strip()
                    aliases = str(row.get("aliases", "")).strip()
                    if code:
                        for a in filter(None, [name] + re.split(r"[,\s/;、，]", aliases)):
                            alias_to_codes[normalize_for_match(a)].add(code)
        except Exception:
            pass
    return alias_to_codes

def load_sentiment_lexicon():
    obj = load_json_if_exists(CONF_DIR / "sentiment_lexicon.json", None)
    if isinstance(obj, dict):
        pos = list({normalize_for_match(w) for w in obj.get("pos", []) if w})
        neg = list({normalize_for_match(w) for w in obj.get("neg", []) if w})
        return pos or DEFAULT_POS, neg or DEFAULT_NEG
    return DEFAULT_POS, DEFAULT_NEG

def load_sentiment_model():
    """
    Load the ML-based sentiment model (voidful/albert_chinese_small_sentiment).
    Returns (tokenizer, model) or (None, None) if not available.
    Uses global cache to avoid reloading.
    """
    global _sentiment_model, _sentiment_tokenizer
    
    if not ML_AVAILABLE:
        return None, None
    
    if _sentiment_model is not None and _sentiment_tokenizer is not None:
        return _sentiment_tokenizer, _sentiment_model
    
    try:
        model_name = "voidful/albert_chinese_small_sentiment"
        print(f"Loading sentiment model: {model_name}...")
        # Use BertTokenizer as recommended for this specific model
        from transformers import BertTokenizer
        _sentiment_tokenizer = BertTokenizer.from_pretrained(model_name)
        _sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _sentiment_model.eval()  # Set to evaluation mode
        print("Sentiment model loaded successfully!")
        return _sentiment_tokenizer, _sentiment_model
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        return None, None

def predict_sentiment_ml(text: str, tokenizer, model):
    """
    Use ML model to predict sentiment of text.
    Returns a score between -1.0 (negative) and +1.0 (positive).
    """
    if tokenizer is None or model is None:
        return 0.0
    
    try:
        # Tokenize and prepare input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Assuming binary classification: [negative, positive]
        # Map to score: -1.0 (negative) to +1.0 (positive)
        neg_prob = probs[0][0].item()
        pos_prob = probs[0][1].item()
        
        # Convert to score: positive - negative
        score = pos_prob - neg_prob
        return score
    except Exception as e:
        print(f"Error in sentiment prediction: {e}")
        return 0.0

# ------------------------------
# Clause analysis & scoring
# ------------------------------
def find_all_positions(text: str, term: str):
    """Yield start indices of 'term' in 'text'."""
    if not term:
        return []
    out = []
    i = text.find(term)
    while i != -1:
        out.append(i)
        i = text.find(term, i + len(term))
    return out

def clause_weights_and_types(sentence: str):
    """
    將一句分割成子句，並對「雖…但…」「即便…仍…」「除非…否則…」等
    以啟發式給予 (clause_text, weight, ctype)：
      - 轉折/讓步 左側（雖/儘管/即便） → 0.6，右側（但/仍/然而） → 1.0
      - 條件（除非…否則…） 左側 → 0.7，右側 → 1.0
      - 若無轉折/條件詞 → 1 個整句，權重 1.0
    """
    s = sentence
    # 先檢查條件（除非/否則）→ 再檢查讓步
    if any(k in s for k in CONDITIONAL_LEFT) and any(k in s for k in CONDITIONAL_RIGHT):
        # 以第一個右側詞切分
        idx = min([i for k in CONDITIONAL_RIGHT if (i := s.find(k)) != -1])
        left = s[:idx]
        right = s[idx:]
        return [(left, 0.7, "conditional-sub"), (right, 1.0, "conditional-main")]

    if any(k in s for k in CONCESSIVE_LEFT) and any(k in s for k in CONCESSIVE_RIGHT):
        idx = min([i for k in CONCESSIVE_RIGHT if (i := s.find(k)) != -1])
        left = s[:idx]
        right = s[idx:]
        return [(left, 0.6, "concessive-sub"), (right, 1.0, "concessive-main")]

    return [(s, 1.0, "plain")]

def score_sentence(sent: str, comp_kp: dict, alias_to_codes: dict, senti_kp: dict, src_w: float = 1.0):
    """
    輕量句級計分：
      - 在每個子句內，找出公司別名與情緒詞的位置。
      - 以視窗長度 WINDOW_CHARS 做對齊：別名附近的情緒詞才會計入。
      - 否定詞若出現在情緒詞 ±6 字元內 → 衰減 (neg_adj=0.6)。
      - 子句權重（轉折/條件）乘到情緒極性。
      - 最後對每個股票彙總 raw，再乘以 src_w 與 neg_adj/clause_w。
    回傳：
      sc: {code: score}、det: [逐句細節]
    """
    sent_n = normalize_for_match(sent)
    pos_words = senti_kp["pos"]
    neg_words = senti_kp["neg"]

    out_sc = defaultdict(float)
    details = []

    for cl_text, w_clause, ctype in clause_weights_and_types(sent_n):
        # 取得該子句中的 alias 與情緒詞位置
        alias_hits = []  # (alias, start_idx, codes)
        for alias, codes in alias_to_codes.items():
            if not alias or alias not in cl_text:
                continue
            for pos in find_all_positions(cl_text, alias):
                alias_hits.append((alias, pos, list(codes)))

        if not alias_hits:
            continue

        pos_hits = []
        neg_hits = []
        for w in pos_words:
            for pos in find_all_positions(cl_text, w):
                pos_hits.append((w, pos))
        for w in neg_words:
            for pos in find_all_positions(cl_text, w):
                neg_hits.append((w, pos))

        # 對齊：以 alias 為中心，視窗內的情緒詞才計分
        for alias, pos_a, codes in alias_hits:
            local = 0.0
            local_details = []
            for word, pos_w in pos_hits:
                if abs(pos_w - pos_a) <= WINDOW_CHARS:
                    # 否定範圍檢查（±6 字元）
                    window = cl_text[max(0, pos_w - 6): pos_w + 6]
                    has_neg = any(tok in window for tok in NEGATION_HINTS)
                    neg_adj = 0.6 if has_neg else 1.0
                    val = +1.0 * w_clause * neg_adj * src_w
                    local += val
                    local_details.append({"word": word, "pol": +1, "neg_adj": neg_adj})
            for word, pos_w in neg_hits:
                if abs(pos_w - pos_a) <= WINDOW_CHARS:
                    window = cl_text[max(0, pos_w - 6): pos_w + 6]
                    has_neg = any(tok in window for tok in NEGATION_HINTS)
                    neg_adj = 0.6 if has_neg else 1.0
                    val = -1.0 * w_clause * neg_adj * src_w
                    local += val
                    local_details.append({"word": word, "pol": -1, "neg_adj": neg_adj})

            if local == 0.0:
                continue

            # 分配到所有對應代碼
            for code in codes:
                out_sc[code] += local
                details.append({
                    "alias": alias,
                    "codes": [code],
                    "ctype": ctype,
                    "clause_w": w_clause,
                    "src_w": src_w,
                    "final": round(local, 4)
                })

    return dict(out_sc), details

def score_sentence_ml(sent: str, comp_kp: dict, alias_to_codes: dict, tokenizer, model, src_w: float = 1.0):
    """
    ML-based sentence scoring:
      - Find company name mentions in the sentence
      - Extract context window around each mention
      - Use ML model to predict sentiment of the context
      - Apply source weight
    Returns:
      sc: {code: score}, det: [details per sentence]
    """
    sent_n = normalize_for_match(sent)
    
    out_sc = defaultdict(float)
    details = []
    
    if tokenizer is None or model is None:
        # Fallback to keyword-based if model not available
        return score_sentence(sent, comp_kp, alias_to_codes, {"pos": DEFAULT_POS, "neg": DEFAULT_NEG}, src_w)
    
    # Find all company mentions in the sentence
    alias_hits = []  # (alias, start_idx, codes)
    for alias, codes in alias_to_codes.items():
        if not alias or alias not in sent_n:
            continue
        for pos in find_all_positions(sent_n, alias):
            alias_hits.append((alias, pos, list(codes)))
    
    if not alias_hits:
        return dict(out_sc), details
    
    # For each company mention, extract context and predict sentiment
    for alias, pos_a, codes in alias_hits:
        # Extract context window around the company mention (±100 chars)
        context_start = max(0, pos_a - 100)
        context_end = min(len(sent_n), pos_a + len(alias) + 100)
        context = sent_n[context_start:context_end]
        
        # Get ML sentiment prediction
        ml_score = predict_sentiment_ml(context, tokenizer, model)
        
        # Apply source weight and scale to match existing score range
        # ML score is [-1, 1], scale to match MAX_PER_ARTICLE_ABS
        final_score = ml_score * src_w * 3.0  # Scale factor to match keyword-based scores
        
        if abs(final_score) < 0.01:  # Skip negligible scores
            continue
        
        # Distribute to all matching company codes
        for code in codes:
            out_sc[code] += final_score
            details.append({
                "alias": alias,
                "codes": [code],
                "method": "ml",
                "ml_score": round(ml_score, 4),
                "src_w": src_w,
                "final": round(final_score, 4)
            })
    
    return dict(out_sc), details


# ------------------------------
# Main
# ------------------------------
def main():
    # 1) Feeds
    conf_feeds = load_json_if_exists(CONF_DIR / "feeds.json", None)
    urls = conf_feeds if isinstance(conf_feeds, list) and conf_feeds else DEFAULT_FEEDS

    # 2) Company aliases & sentiment lexicon
    alias_to_codes = load_company_alias()
    # comp_kp 保留接口（目前未用到，預留進階規則）
    comp_kp = {}
    pos_lex, neg_lex = load_sentiment_lexicon()
    senti_kp = {"pos": [normalize_for_match(w) for w in pos_lex],
                "neg": [normalize_for_match(w) for w in neg_lex]}

    # 3) Source weights
    src_weights = load_source_weights()
    
    # 4) Load ML sentiment model
    print("Initializing sentiment analysis model...")
    tokenizer, model = load_sentiment_model()
    if tokenizer and model:
        print("Using ML-based sentiment analysis")
    else:
        print("Using keyword-based sentiment analysis (fallback)")

    # 5) Load existing records to avoid duplicates
    # Strategy: Read all daily files to build seen_ids.
    # Also read legacy news_compact.json if it exists and migrate it.
    all_news = []
    seen_ids = set()

    # Load from daily files
    for f in NEWS_DIR.glob("*.json"):
        records = load_json_if_exists(f, [])
        for r in records:
            if r.get("id") and r.get("id") not in seen_ids:
                seen_ids.add(r["id"])
                all_news.append(r)



    added = 0
    with NEWS_LOG.open("a", encoding="utf-8") as logf:
        for u in urls:
            feed = feedparser.parse(u)
            entries = feed.entries or []
            for e in entries:
                title = (e.get("title") or "").strip()
                summary = (e.get("summary") or e.get("description") or "").strip()
                link = (e.get("link") or "").strip()
                raw = f"{title}\n{summary}"
                text = BeautifulSoup(raw, "html.parser").get_text()
                text = normalize_for_match(text)

                # 時間
                ts = None
                for key in ("published", "updated"):
                    if e.get(key):
                        try:
                            ts = datetime.fromtimestamp(time.mktime(e.get(key + "_parsed")), tz=timezone.utc)
                            break
                        except Exception:
                            try:
                                ts = datetime.fromisoformat(e.get(key))
                                break
                            except Exception:
                                pass
                if ts is None:
                    ts = datetime.now(timezone.utc)

                # 來源 host
                try:
                    source_host = urlparse(link).netloc or ""
                except Exception:
                    source_host = ""

                # 來源權重（正負分在 score_sentence 內決定；這裡以 1.0 打底）
                src_w = 1.0  # 給 score_sentence 用；若你要直接整體加權，可把 sw 放進該函數
                # 切句
                sent_list = split_sentences(text, SENT_SPLIT)

                per_company = {}
                details_all = []
                art_pos = 0.0
                art_neg = 0.0

                for sent in sent_list:
                    # Use ML-based scoring if model is available, otherwise fallback to keyword-based
                    if tokenizer and model:
                        sc, det = score_sentence_ml(sent, comp_kp, alias_to_codes, tokenizer, model, src_w=src_w)
                    else:
                        sc, det = score_sentence(sent, comp_kp, alias_to_codes, senti_kp, src_w=src_w)
                    # 聚合公司分數
                    for k, v in sc.items():
                        per_company[k] = per_company.get(k, 0.0) + float(v)

                    # 累積逐句明細（容錯）
                    det = det or []
                    for d in det:
                        d.setdefault("ctype", "unknown")
                        d.setdefault("clause_w", 1.0)
                        d.setdefault("neg_adj", 1.0)
                        d.setdefault("src_w", src_w)
                        # 文章層級情緒累計（final 為單句對單股貢獻，這裡只計總強度）
                        val = float(d.get("final", 0.0) or 0.0)
                        if val >= 0:
                            art_pos += val
                        else:
                            art_neg += val
                    details_all.extend(det)

                # 單文檔上限截斷（避免極端）
                per_company = cap_scores(per_company, MAX_PER_ARTICLE_ABS)

                # 建立新聞 ID
                nid = make_news_id(title, link)
                if nid in seen_ids:
                    continue

                rec = {
                    "id": nid,
                    "ts": ts.isoformat(),
                    "title": title,
                    "link": link,
                    "source_host": source_host,            # 前端直接使用
                    "codes": sorted(per_company.keys()),   # 可能為空，仍保存
                    "per_company": per_company,            # 可能為空
                    "sent_pos": round(art_pos, 3),
                    "sent_neg": round(art_neg, 3),
                    "sent_score": round(art_pos + art_neg, 3),
                    "detail": details_all
                }

                # 一律保存
                all_news.append(rec)
                logf.write(json.dumps({
                    "ts": rec["ts"],
                    "title": title,
                    "link": link,
                    "per_company_raw": per_company,
                    "sent_score": rec["sent_score"],
                    "source_host": source_host
                }, ensure_ascii=False) + "\n")
                added += 1
                seen_ids.add(nid)

    # 5) Persist news split by day
    # Group by date
    news_by_date = defaultdict(list)
    for n in all_news:
        if not isinstance(n, dict) or not n.get("ts"):
            continue
        try:
            # Parse TS to get YYYY-MM-DD
            # Handle both Z and +00:00
            ts_str = n["ts"].replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts_str)
            date_str = dt.date().isoformat()
            news_by_date[date_str].append(n)
        except Exception:
            continue

    # Write daily files
    available_dates = []
    for date_str, items in news_by_date.items():
        # Sort items by TS within the day
        items.sort(key=lambda x: x.get("ts", ""))
        p = NEWS_DIR / f"{date_str}.json"
        write_json(p, items)
        available_dates.append(date_str)

    # Write today.json (Taiwan date, newest to oldest)
    tz_tw = timezone(timedelta(hours=8))
    today_tw = datetime.now(tz_tw).date()
    
    today_items = []
    for n in all_news:
        try:
            ts_str = n["ts"].replace("Z", "+00:00")
            dt_utc = datetime.fromisoformat(ts_str)
            dt_tw = dt_utc.astimezone(tz_tw)
            if dt_tw.date() == today_tw:
                today_items.append(n)
        except Exception:
            pass

    write_json(TODAY_JSON, sorted(today_items, key=lambda x: x.get("ts", ""), reverse=True))
    
    # Write index file
    available_dates.sort(reverse=True) # Newest first
    write_json(NEWS_INDEX, available_dates)

    # 6) Daily sentiment aggregation for sparkline
    agg = defaultdict(lambda: defaultdict(float))  # agg[code][YYYY-MM-DD] = sum_score
    for n in all_news:
        ts = n.get("ts")
        try:
            day = datetime.fromisoformat(ts.replace("Z", "+00:00")).date().isoformat()
        except Exception:
            day = (datetime.utcnow().date()).isoformat()
        s = float(n.get("sent_score", 0.0) or 0.0)
        for c in (n.get("codes") or []):
            agg[c][day] += s

    daily_rows = []
    for c, days in agg.items():
        for day, val in sorted(days.items()):
            daily_rows.append({"code": c, "date": day, "score": round(val, 3)})
    write_json(DAILY_JSON, daily_rows)

    print(f"[fetch_and_rank] added={added}, total_records={len(all_news)}")


if __name__ == "__main__":
    main()
