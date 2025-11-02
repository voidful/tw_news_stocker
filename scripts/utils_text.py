# -*- coding: utf-8 -*-
import re
from typing import List

# 基礎正規化（僅移除空白與全形空格）
def normalize_for_match(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\u3000", "").replace(" ", "")
    # 移除零寬與常見控制符
    t = re.sub(r"[\u200b\ufeff\r\t]", "", t)
    return t

def split_sentences(text: str, seps: str) -> List[str]:
    if not text:
        return []
    # 以任一分隔符切句
    pattern = "[" + re.escape(seps) + "]"
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]
