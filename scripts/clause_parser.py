# -*- coding: utf-8 -*-
# Lightweight clause decomposition for concessive / contrastive / conditional Chinese patterns.

import re
from typing import List, Tuple

def find_last_any(s: str, toks: list) -> int:
    pos = -1
    for t in toks:
        p = s.rfind(t)
        if p > pos:
            pos = p
    return pos

def decompose_clauses(s: str, concessive_prefix: list, contrastive: list, persistive: list, cond_if: list, cond_then: list):
    """
    Return list of (clause_text, weight) where weights encode discourse salience.
    Rules:
      - concessive + contrast/persist: left (0.5), right (1.1)
      - conditional: antecedent (0.7), consequent (1.1)
      - fallback: whole s with weight 1.0
    """
    # normalize thin spaces
    s = s.replace("\u200b","").replace("\ufeff","")
    # Concessive
    has_pref = any(k in s for k in concessive_prefix)
    has_contr = any(k in s for k in contrastive) or any(k in s for k in persistive)
    if has_pref and has_contr:
        cut = max([s.rfind(k) for k in contrastive + persistive if k in s] + [-1])
        if cut >= 0:
            left = s[:cut]
            right = s[cut:]
            return [(left, 0.5), (right, 1.1)]

    # Conditional
    has_if = any(k in s for k in cond_if)
    has_then = any(k in s for k in cond_then)
    if has_if and has_then:
        cut = max([s.rfind(k) for k in cond_then if k in s] + [-1])
        if cut >= 0:
            left = s[:cut]
            right = s[cut:]
            return [(left, 0.7), (right, 1.1)]

    return [(s, 1.0)]


def decompose_except_if(s: str, ex_if: list, ex_else: list, neg_tokens: list=None):
    has_if = any(k in s for k in ex_if)
    has_else = any(k in s for k in ex_else)
    if has_if and has_else:
        cut = max([s.rfind(k) for k in ex_else if k in s] + [-1])
        if cut >= 0:
            left = s[:cut]
            right = s[cut:]
            w_left, w_right = 0.4, 1.2
            if neg_tokens:
                if any(t in left for t in neg_tokens):
                    w_left, w_right = 0.3, 1.25
            return [(left, w_left, {"except":"left"}), (right, w_right, {"except":"right"})]
    return None

