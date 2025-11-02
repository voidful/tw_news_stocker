#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from flashtext import KeywordProcessor

from scripts.company_sources import fetch_companies, normalize_name
from scripts.keywords import POS_KWS, NEG_KWS, NEGATION_HINTS, SENT_SPLIT
from scripts.utils_text import normalize_for_match, split_sentences
from scripts.clause_parser import decompose_clauses, decompose_except_if
from scripts.supply_loader import load_relations

# ---------------- Config ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CONF_DIR = ROOT / "config"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPANY_CACHE = DATA_DIR / "companies.json"
NEWS_LOG = DATA_DIR / "news_log.jsonl"

WINDOWS = [1,3,5,10,30,60]
TZ = ZoneInfo(os.environ.get("TZ", "Asia/Taipei"))
MAX_PER_ARTICLE_ABS = 2          # 單新聞對單公司分數上限
NEAR_CHARS = 24                   # 公司與情緒詞距離窗（字元數）
NEGATION_LOOKBACK = 3             # 否定詞出現在情緒詞前多少字元內視為反轉

# ---------------- RSS sources ----------------
def load_rss_sources() -> list:
    env = (os.environ.get("RSS_URL") or "").strip()
    urls = []
    if env:
        urls.append(env)
    conf = CONF_DIR / "rss_sources.txt"
    if conf.exists():
        for line in conf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    # 去重
    out, seen = [], set()
    for u in urls:
        if u not in seen:
            out.append(u); seen.add(u)
    return out or [
        # fallback: Google News 台股/台灣科技
        "https://news.google.com/rss/search?q=%E5%8F%B0%E8%82%A1&hl=zh-TW&gl=TW&ceid=TW%3Azh-Hant"
    ]

# ---------------- Utils ----------------
def http_get_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent":"tw-stock-news-bot/1.0"})
        r.raise_for_status()
        return r.text
    except Exception:
        return ""

def parse_pub_time(entry):
    try:
        if entry.get("published_parsed"):
            ts = time.mktime(entry["published_parsed"])
            return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(TZ)
    except Exception:
        pass
    return datetime.now(TZ)

def make_news_id(title, link):
    raw = f"{title}::{link}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()[:16]

# ---------------- Build keyword processors ----------------
def build_company_processors(companies: list):
    # 映射 alias -> set(codes)
    alias_to_codes = {}
    for c in companies:
        for a in c["aliases"]:
            if len(a) >= 2:
                alias_to_codes.setdefault(a, set()).add(c["code"])

    # FlashText 只能映射到單值；我們先映射到 alias 字串本身，再由 alias_to_codes 還原代碼集合
    kp = KeywordProcessor(case_sensitive=True)  # 中文區分大小寫無關
    for alias in alias_to_codes.keys():
        kp.add_keyword(alias, alias)  # value=alias

    return kp, alias_to_codes

def build_sentiment_processor():
    kp = KeywordProcessor(case_sensitive=True)
    for w in POS_KWS:
        kp.add_keyword(w, ("POS", w))
    for w in NEG_KWS:
        kp.add_keyword(w, ("NEG", w))
    return kp

# ---------------- Scoring ----------------
def apply_negation(text: str, start_idx: int, label: str) -> str:
    # 若在 start_idx 前 NEGATION_LOOKBACK 範圍內有否定提示詞，則反轉
    l = max(0, start_idx - NEGATION_LOOKBACK)
    window = text[l:start_idx]
    for hint in NEGATION_HINTS:
        if hint in window:
            return "NEG" if label=="POS" else "POS"
    return label

def score_sentence(sent: str, comp_kp, alias_to_codes, senti_kp):
    # 回傳：dict code -> score 以及命中細節
    res = {}
    # 提前正規化，移除空格以穩定距離度量
    s = normalize_for_match(sent)

    # span_info 以便計算距離
    comp_hits = comp_kp.extract_keywords(s, span_info=True)  # [(alias, start, end), ...]
    senti_hits = senti_kp.extract_keywords(s, span_info=True)  # [((label,word), start, end), ...]

    if not comp_hits or not senti_hits:
        return res, []

    details = []
    # 將每個公司別名與情緒詞做近鄰配對
    for alias, a_start, a_end in comp_hits:
        codes = alias_to_codes.get(alias, set())
        a_mid = (a_start + a_end) // 2
        for (label, word), s_start, s_end in senti_hits:
            s_mid = (s_start + s_end) // 2
            dist = abs(a_mid - s_mid)
            if dist <= NEAR_CHARS:
                eff_label = apply_negation(s, s_start, label)
                val = 1 if eff_label == "POS" else -1
                for code in codes:
                    res[code] = res.get(code, 0) + val
                details.append({"alias": alias, "codes": list(codes), "word": word, "label": label, "eff_label": eff_label, "dist": dist})
    return res, details

def cap_scores(per_company: dict, cap_abs: int):
    out = {}
    for k,v in per_company.items():
        if v > cap_abs: v = cap_abs
        if v < -cap_abs: v = -cap_abs
        out[k] = v
    return out

# ---------------- Aggregation ----------------
def aggregate_leaderboards(now: datetime, companies_map: dict, log_path: Path):
    rows = []
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])

    out = {}
    for d in WINDOWS:
        start = now - timedelta(days=d)
        sub = df[(df["ts"] >= start) & (df["ts"] <= now)]
        if sub.empty:
            out[d] = pd.DataFrame(columns=["rank","code","name","score","hits"])
            continue

        sub = sub.explode("codes")
        sub = sub[sub["codes"].notna()]

        agg = sub.groupby("codes").agg(
            score=("score","sum"),
            hits=("score","count")
        ).reset_index().rename(columns={"codes":"code"})
        agg["name"] = agg["code"].map(lambda x: companies_map.get(x, x))
        agg = agg.sort_values(["score","hits"], ascending=[False, False]).reset_index(drop=True)
        agg.insert(0, "rank", range(1, len(agg)+1))
        out[d] = agg

        agg.to_csv(OUT_DIR / f"leaderboard_{d}d.csv", index=False, encoding="utf-8-sig")

    # Markdown
    md_lines = ["# 台股新聞情緒排行榜（規則版，自動產生）", f"- 產生時間：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}", ""]
    for d in WINDOWS:
        md_lines.append(f"## {d} 天")
        dfw = out[d].head(30)
        if dfw.empty:
            md_lines.append("_目前無資料_"); md_lines.append(""); continue
        md_lines.append("| 排名 | 代碼 | 名稱 | 分數 | 條數 |")
        md_lines.append("|---:|---:|---|---:|---:|")
        for _, r in dfw.iterrows():
            md_lines.append(f"| {int(r['rank'])} | {r['code']} | {r['name']} | {int(r['score'])} | {int(r['hits'])} |")
        md_lines.append("")
    (OUT_DIR / "leaderboards.md").write_text("\n".join(md_lines), encoding="utf-8")

    return out

# ---------------- Main ----------------
def main():
    now = datetime.now(TZ)

    # 1) 公司表
    companies = fetch_companies(COMPANY_CACHE)
    companies_map = {c["code"]: c["name"] for c in companies}
    comp_kp, alias_to_codes = build_company_processors(companies)
    senti_kp = build_sentiment_processor()

    # 2) RSS 來源
    urls = load_rss_sources()
    seen_ids = set()
    if NEWS_LOG.exists():
        with NEWS_LOG.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen_ids.add(rec.get("id"))
                except Exception:
                    continue

    added = 0
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

            # 切句，逐句評分
            sent_list = split_sentences(text, SENT_SPLIT)

            per_company = {}
            details_all = []
            for sent in sent_list:
                sc, det = score_sentence(sent, comp_kp, alias_to_codes, senti_kp)
                for k,v in sc.items():
                    per_company[k] = per_company.get(k, 0) + v
                for d in det:
            if "ctype" not in d:
                d["ctype"] = meta.get("type","unknown") if isinstance(meta, dict) else "unknown"
        details_all.extend(det)

            per_company = cap_scores(per_company, MAX_PER_ARTICLE_ABS)
            if not per_company:
                continue

            nid = make_news_id(title, link)
            if nid in seen_ids:
                continue

            # 最終簽入
            score_sum = sum(per_company.values())
            codes_sorted = sorted(per_company.keys())
            rec = {
                "id": nid,
                "ts": parse_pub_time(e).strftime("%Y-%m-%dT%H:%M:%S%z"),
                "title": title,
                "link": link,
                "score": int(score_sum),
                "codes": codes_sorted,
                "detail": details_all[:20]  # 控制檔案體積
            }
            with NEWS_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            added += 1

    # 3) Leaderboards
    aggregate_leaderboards(now, companies_map, NEWS_LOG)

    print(f"新增新聞記錄：{added} 筆")

if __name__ == "__main__":
    main()
