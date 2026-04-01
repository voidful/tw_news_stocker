# -*- coding: utf-8 -*-
import json
from pathlib import Path

def load_relations(path_relations: Path, path_weights: Path):
    rel = {"subsidiaries": {}, "parents": {}, "suppliers": {}, "customers": {}}
    w = {"subsidiary_to_parent": 1.0, "parent_to_subsidiary": 0.8, "customer_to_supplier": 0.5, "supplier_to_customer": 0.3}
    try:
        rel.update(json.loads(path_relations.read_text(encoding="utf-8")))
    except Exception:
        pass
    try:
        w.update(json.loads(path_weights.read_text(encoding="utf-8")))
    except Exception:
        pass
    # auto inverse relationships if missing
    if not rel.get("parents") and rel.get("subsidiaries"):
        parents = {}
        for p, subs in rel["subsidiaries"].items():
            for s in subs:
                parents.setdefault(s, []).append(p)
        rel["parents"] = parents
    if not rel.get("customers") and rel.get("suppliers"):
        customers = {}
        for c, sups in rel["suppliers"].items():
            for s in sups:
                customers.setdefault(s, []).append(c)
        rel["customers"] = customers
    return rel, w
