# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
from flashtext import KeywordProcessor
import math
import os
from scripts.company_sources import fetch_companies
from scripts.brand_map import load_brand_aliases
try:
    from scripts.keywords import POSITIVE, NEGATIVE
    POS_WORDS = POSITIVE
    NEG_WORDS = NEGATIVE
except Exception:
    POS_WORDS = ["看好","利多","成長","擴產","創新高","上修"]
    NEG_WORDS = ["看淡","利空","衰退","減產","下修","下跌"]


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONF_DIR = ROOT / "config"
NEWS_LOG = DATA_DIR / "news_log.jsonl"

PATTERNS = [
    (r"(?P<a>.+?)(合作|攜手|結盟|簽約|MOU|聯手)(?P>b>.+)", "partner", 1.0, None),
    (r"(?P<a>.+?)(供應|代工|出貨給|導入至|採用)(?P<b>.+)", "supplier_to_customer", 1.2, "a->b"),
    (r"(?P<a>.+?)(成為|列為)(?P<b>.+?)(供應商|供應鏈)", "supplier_to_customer", 1.1, "a->b"),
    (r"(?P<a>.+?)(為|代工)(?P<b>.+?)(產品|手機|筆電|車用|伺服器)", "supplier_to_customer", 1.0, "a->b"),
    (r"(?P<a>.+?)(客戶|訂單來自)(?P<b>.+)", "customer_to_supplier", 1.1, "a->b"),
]

def build_alias_index(companies, brand_map):
    kp = KeywordProcessor(case_sensitive=True)
    alias2codes = {}
    for c in companies:
        for a in c.get("aliases", []):
            if len(a) >= 2:
                alias2codes.setdefault(a, set()).add(c["code"])
                kp.add_keyword(a, a)
    for a, code in brand_map.items():
        alias2codes.setdefault(a, set()).add(code)
        kp.add_keyword(a, a)
    return kp, alias2codes

def align(text, span):
    # 字距對齊：返回該片段內別名命中清單與中心位置
    alias, s, e = span
    center = (s+e)/2
    return alias, s, e, center

USE_STANZA = os.getenv("USE_STANZA","0") == "1"
try:
    if USE_STANZA:
        import stanza
        STZ_NLP = stanza.Pipeline("zh-hant", processors="tokenize,pos,depparse", tokenize_no_ssplit=True, use_gpu=False)
    else:
        STZ_NLP = None
except Exception:
    STZ_NLP = None

def dep_path_len(text, a_span, b_span):
    if STZ_NLP is None:
        return None
    doc = STZ_NLP(text)
    sent = doc.sentences[0] if doc.sentences else None
    if not sent: return None
    # map char offsets to token indices
    def tok_idx_at(char_pos):
        for i,tok in enumerate(sent.tokens):
            if tok.start_char is not None and tok.end_char is not None and tok.start_char<=char_pos<tok.end_char:
                return i
        return None
    ai = tok_idx_at(a_span[0]); bi = tok_idx_at(b_span[0])
    if ai is None or bi is None: return None
    # build undirected graph via heads
    n = len(sent.words)
    g = {i: set() for i in range(n)}
    for i,w in enumerate(sent.words):
        if w.head>0:
            g[i].add(w.head-1); g[w.head-1].add(i)
    # BFS shortest path
    from collections import deque
    vis={ai}; q=deque([(ai,0)])
    while q:
        u,d=q.popleft()
        if u==bi: return d
        for v in g.get(u,[]):
            if v not in vis:
                vis.add(v); q.append((v,d+1))
    return None

def local_polarity(text, span_start, span_end, window=12):
(text, span_start, span_end, window=12):
    left = max(0, span_start-window); right = min(len(text), span_end+window)
    seg = text[left:right]
    pos = any(w in seg for w in POS_WORDS); neg = any(w in seg for w in NEG_WORDS)
    if pos and not neg: return 1
    if neg and not pos: return -1
    return 0

def extract_pairs(text, kp, alias2codes):
    hits = kp.extract_keywords(text, span_info=True)
    spans = [(alias, s, e) for alias, s, e in hits]
    res = []
    for pat, reltype in PATTERNS:
        m = re.search(pat, text)
        if not m:
            continue
        la, lb = m.group("a"), m.group("b")
        a_s, a_e = text.find(la), text.find(la)+len(la)
        b_s, b_e = text.find(lb), text.find(lb)+len(lb)
        # find aliases inside la/lb
        A = [(alias, s, e) for alias, s, e in spans if s>=a_s and e<=a_e]
        B = [(alias, s, e) for alias, s, e in spans if s>=b_s and e<=b_e]
        codesA = set(); codesB = set()
        for a in A: codesA.update(alias2codes.get(a[0], set()))
        for b in B: codesB.update(alias2codes.get(b[0], set()))
        # roles: subject/object by pattern (a then b)
        # sentiment near each side
        polA = 0; polB = 0
        if A: polA = local_polarity(text, A[0][1], A[0][2])
        if B: polB = local_polarity(text, B[0][1], B[0][2])
        for ca in codesA:
            for cb in codesB:
                if ca == cb: continue
                # distance and base score
                dist = abs(a_s - b_s)
                base = 1.0/(1.0 + dist/10.0)
                # align polarity bonus
                bonus = 1.0 + 0.1*(abs(polA) + abs(polB))
                score = base * bonus
                impact_subj = 1 if polA>0 else (-1 if polA<0 else 0)
impact_obj  = 1 if polB>0 else (-1 if polB<0 else 0)
# rule: in supplier_to_customer, subj=供應商、obj=客戶；正面常同向；負面亦常同向（供應拉貨/砍單）
if reltype=="supplier_to_customer":
    pass
elif reltype=="customer_to_supplier":
    # 客戶端正面（需求好）→ 對供應商通常也正面；負面亦然
    if impact_obj!=0 and impact_subj==0:
        impact_subj = impact_obj
res.append({"subj": ca, "obj": cb, "rel": reltype, "score": score, "pol_subj": polA, "pol_obj": polB, "impact_subj": impact_subj, "impact_obj": impact_obj})
    return res



def main():
    companies = fetch_companies(DATA_DIR / "companies.json")
    brand_map = load_brand_aliases(CONF_DIR / "brand_aliases.json")
    kp, alias2codes = build_alias_index(companies, brand_map)

    rows = []
    if not NEWS_LOG.exists():
        print("no news_log"); return
    for line in NEWS_LOG.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except:
            continue
        title = (rec.get("title") or "").replace(" ", "")
        pairs = extract_pairs(title, kp, alias2codes)
        rows.extend(pairs)
    # aggregate with weight sum
    from collections import defaultdict
    agg = defaultdict(float)
    for r in rows:
        agg[(r["a"], r["b"], r["rel"])] += r["w"]
    out = [{"a": a, "b": b, "rel": rel, "score": round(w,3)} for (a,b,rel), w in agg.items() if w >= 1.0]
    (CONF_DIR / "relations_suggested.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"suggested edges: {len(out)}")

if __name__ == "__main__":
    main()
