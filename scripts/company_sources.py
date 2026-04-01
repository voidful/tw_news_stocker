# -*- coding: utf-8 -*-
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from bs4 import BeautifulSoup

CLEAN_PATTERNS = [
    r"股份有限公司", r"有限公司", r"\(股\)公司", r"\(股\)", r"公司", r"　", r"\s"
]
CLEAN_REGEX = re.compile("|".join(CLEAN_PATTERNS))

def normalize_name(name: str) -> str:
    if not name:
        return ""
    n = CLEAN_REGEX.sub("", name)
    n = re.sub(r"[（）()《》〈〉【】\[\]「」『』．·.、，,。!！?？\-–—_~／/＋+&＆:：；;]", "", n)
    return n.strip()

def http_get_json_candidates(candidates):
    s = requests.Session()
    s.headers.update({"User-Agent": "tw-stock-news-bot/1.0"})
    for url in candidates:
        try:
            r = s.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return v
                return [data]
            if isinstance(data, list):
                return data
        except Exception:
            continue
    return []

def http_get_text(url):
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent":"tw-stock-news-bot/1.0"})
        r.raise_for_status()
        return r.text
    except Exception:
        return ""

def extract_code_and_name(row: dict):
    code_keys = ["公司代號","證券代號","stock_code","Code","SecuritiesCode","stockNo","公司代碼"]
    name_keys = ["公司名稱","公司簡稱","證券名稱","Name","中文簡稱","公司全名","ShortName","chineseName"]
    code, name = None, None
    for k in code_keys:
        if k in row and isinstance(row[k], str):
            m = re.search(r"\d{4,5}", row[k])
            if m:
                code = m.group(0)
                break
    for k in name_keys:
        if k in row and isinstance(row[k], str) and re.search(r"[\u4e00-\u9fff]", row[k]):
            name = row[k].strip()
            break
    return code, name

def fetch_companies(cache_path: Path) -> List[dict]:
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and len(cached) > 1000:
                return cached
        except Exception:
            pass

    twse_candidates = [
        "https://openapi.twse.com.tw/v1/opendata/t187ap03_L",
        "https://mopsfin.twse.com.tw/opendata/t187ap03_L",
    ]
    tpex_candidates = [
        "https://www.tpex.org.tw/openapi/v1/company_basic_info",
        "https://mopsfin.twse.com.tw/opendata/t187ap03_O",
        "https://mopsfin.twse.com.tw/opendata/t187ap14_O",
    ]

    twse = http_get_json_candidates(twse_candidates)
    tpex = http_get_json_candidates(tpex_candidates)

    if len(twse) < 100:
        html = http_get_text("https://www.twse.com.tw/zh/listed/companies")
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        for tr in soup.select("table tr"):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) >= 2 and re.fullmatch(r"\d{4,5}", tds[0]):
                rows.append({"公司代號": tds[0], "公司名稱": tds[1]})
        if rows:
            twse = rows

    companies = []
    for src in (twse or []) + (tpex or []):
        code, name = extract_code_and_name(src)
        if code and name:
            companies.append({"code": code, "name": name})

    uniq = {}
    for c in companies:
        uniq[c["code"]] = c
    companies = list(uniq.values())

    for c in companies:
        base = c["name"]
        aliases = {normalize_name(base)}
        parts = re.split(r"[（）()]", base)
        for p in parts:
            p = p.strip()
            if p and re.search(r"[\u4e00-\u9fff]", p):
                aliases.add(normalize_name(p))
        aliases.add(normalize_name(re.sub(r"-KY$", "", base, flags=re.IGNORECASE)))
        c["aliases"] = sorted({a for a in aliases if a})

    try:
        cache_path.write_text(json.dumps(companies, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return companies
