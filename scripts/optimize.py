# -*- coding: utf-8 -*-
import json, numpy as np, pandas as pd
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CONF_DIR = ROOT / "config"
NEWS_LOG = DATA_DIR / "news_log.jsonl"

CFG = json.loads((CONF_DIR / "optimize.json").read_text(encoding="utf-8"))

def parse_host(u):
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""

def load_prices(codes):
    prices = {}
    for code in set(codes):
        try:
            url = f"https://raw.githubusercontent.com/voidful/tw_stocker/main/data/{code}.csv"
            df = pd.read_csv(url)
            ts = pd.to_datetime(df["Datetime"] if "Datetime" in df.columns else df["date"])
            close = df.get("close", df.get("Close", df.get("c")))
            s = pd.Series(close.values, index=ts).resample("1D").last().dropna()
            prices[code] = s
        except Exception:
            continue
    return prices

def time_decay_weight(age_days: float, half_life_days: float) -> float:
    return 2.0 ** (-age_days / half_life_days) if half_life_days>0 else 1.0

def daily_sharpe(scores, prices, topN=5, min_score=1.0):
    if scores.empty: return 0.0
    dates = pd.date_range(scores["ts"].min().date()+pd.Timedelta(days=1), scores["ts"].max().date(), freq="B")
    rets = []
    for d in dates:
        cutoff = pd.Timestamp(d).normalize()-pd.Timedelta(days=1)
        sub = scores[scores["ts"] <= cutoff]
        if sub.empty:
            rets.append(0.0); continue
        rank = sub.groupby("code")["sig"].sum().sort_values(ascending=False)
        picks = rank[rank>=min_score].head(topN).index.tolist()
        vals = []
        for c in picks:
            s = prices.get(c)
            if s is None or d not in s.index or (d+pd.Timedelta(days=1)) not in s.index: continue
            vals.append(float(s.loc[d+pd.Timedelta(days=1)]/s.loc[d]-1.0))
        rets.append(np.mean(vals) if vals else 0.0)
    r = pd.Series(rets)
    return float(r.mean()/(r.std()+1e-12)*(252**0.5))

def load_groups():
    path = CONF_DIR / "source_groups.json"
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}

def host_to_group(host: str, groups: dict):
    if host in groups:
        return groups[host]
    parts = (host or "").split(".")
    if len(parts)>=2:
        return ".".join(parts[-2:])
    return host or ""

def ts_kfold_ranges(start, end, k=3, train_months=12, valid_months=3, repeat=1):
    all_ranges = []
    for rep in range(repeat):
        ranges = []
        cur_valid_end = end - pd.DateOffset(months=rep*valid_months)
        for i in range(k):
            valid_end = cur_valid_end - pd.DateOffset(months=i*valid_months)
            valid_start = valid_end - pd.DateOffset(months=valid_months)
            train_end = valid_start - pd.Timedelta(days=1)
            train_start = train_end - pd.DateOffset(months=train_months)
            ranges.append((train_start, train_end, valid_start, valid_end))
        all_ranges.append(list(reversed(ranges)))
    return all_ranges

def main():
    rows = []
    if not NEWS_LOG.exists():
        raise SystemExit("news_log.jsonl not found.")
    for line in NEWS_LOG.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except:
            pass
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    recs = []
    for _, r in df.iterrows():
        ts = r["ts"]; link = r.get("link",""); host = parse_host(link)
        pc = r.get("per_company_raw") or r.get("per_company") or {}
        for code, val in (pc.items() if isinstance(pc, dict) else []):
            recs.append({"ts": ts, "host": host, "code": code, "raw": float(val)})
    ds = pd.DataFrame(recs)
    if ds.empty:
        raise SystemExit("No per_company_raw found; run new pipeline to populate.")
    prices = load_prices(ds["code"].unique())

    end_all = ds["ts"].max()
    start_all = ds["ts"].min()

    k = int(CFG.get("k_folds", 3))
    repeats = int(CFG.get("k_repeats", 1))
    ranges_all = ts_kfold_ranges(start_all, end_all, k=k, train_months=CFG["train_months"], valid_months=CFG["valid_months"], repeat=repeats)

    # Half-life selection across repeats
    hl_scores = []
    best_hl, best_val = None, -1e9
    for hl in CFG.get("hl_grid", [3,5,7,10,14,21,30]):
        tmp = ds.copy()
        tmp["age_days"] = (tmp["ts"].max() - tmp["ts"]).dt.total_seconds()/86400.0
        tmp["w"] = tmp["age_days"].apply(lambda x: time_decay_weight(x, hl))
        tmp["sig"] = tmp["raw"] * tmp["w"]
        vals = []
        for ranges in ranges_all:
            for (tr_s, tr_e, va_s, va_e) in ranges:
                va = tmp[(tmp["ts"]>=va_s) & (tmp["ts"]<=va_e)]
                vals.append(daily_sharpe(va, prices, topN=CFG["topN"], min_score=CFG["min_score"]))
        mean_v = float(np.mean(vals)) if vals else -1e9
        hl_scores.append({"half_life": hl, "cv_sharpe": mean_v})
        if mean_v > best_val:
            best_val, best_hl = mean_v, hl
    pd.DataFrame(hl_scores).to_csv(OUT_DIR / "half_life_cv.csv", index=False, encoding="utf-8-sig")

    # Group weights + λ via one-standard-error rule
    groups_map = load_groups()
    ds["group"] = ds["host"].apply(lambda h: host_to_group(h, groups_map))
    groups = ds["group"].value_counts().index.tolist()
    lam_grid = CFG.get("lambda_grid", [0.0,0.02,0.05,0.1,0.2])
    coarse_grid = CFG.get("host_grid", [0.8,1.0,1.2,1.4])

    def evaluate_weights(wg):
        tmp = ds.copy()
        tmp["sw"] = tmp["group"].apply(lambda x: wg.get(x,1.0))
        tmp["age_days"] = (tmp["ts"].max() - tmp["ts"]).dt.total_seconds()/86400.0
        tmp["w"] = tmp["age_days"].apply(lambda x: time_decay_weight(x, best_hl))
        tmp["sig"] = tmp["raw"] * tmp["sw"] * tmp["w"]
        scores = []
        for ranges in ranges_all:
            fold_vals = []
            for (_, _, va_s, va_e) in ranges:
                va = tmp[(tmp["ts"]>=va_s) & (tmp["ts"]<=va_e)]
                fold_vals.append(daily_sharpe(va, prices, topN=CFG["topN"], min_score=CFG["min_score"]))
            scores.append((np.mean(fold_vals), np.std(fold_vals)/np.sqrt(max(1,len(fold_vals)))))
        mean_cv = float(np.mean([m for (m, se) in scores]))
        se_cv = float(np.sqrt(np.mean([se**2 for (m, se) in scores])))
        return mean_cv, se_cv

    # warm start λ=0
    wg = {g:1.0 for g in groups}
    for _ in range(2):
        base_score, _ = evaluate_weights(wg)
        improved = False
        for g in groups:
            best_v = wg[g]; best_s = -1e9
            for c in coarse_grid:
                cand = dict(wg); cand[g] = c
                s, _ = evaluate_weights(cand)
                if s > best_s:
                    best_s, best_v = s, c
            if best_s > base_score:
                wg[g] = best_v; base_score = best_s; improved = True
        if not improved: break

    # λ grid with one-standard-error rule
    records = []
    for lam in lam_grid:
        w = dict(wg)
        # quick coordinate search with penalty
        for _ in range(2):
            for g in groups:
                best_v = w[g]; best_obj = 1e9; best_mean = -1e9; best_se = 0.0
                for c in coarse_grid:
                    cand = dict(w); cand[g] = c
                    mean_cv, se_cv = evaluate_weights(cand)
                    l1 = sum(abs(v-1.0) for v in cand.values())
                    obj = -(mean_cv) + lam * l1
                    if obj < best_obj:
                        best_obj, best_v, best_mean, best_se = obj, c, mean_cv, se_cv
                w[g] = best_v
        l1 = sum(abs(v-1.0) for v in w.values())
        records.append({"lambda": lam, "mean": best_mean, "se": best_se, "l1": l1, "weights": w})

    best_idx = max(range(len(records)), key=lambda i: records[i]["mean"])
    best_mean = records[best_idx]["mean"]
    best_se = records[best_idx]["se"]
    threshold = best_mean - best_se
    candidates = [r for r in records if r["mean"] >= threshold]
    if candidates:
        chosen = sorted(candidates, key=lambda r: (-(r["lambda"]), r["l1"]))[0]
    else:
        chosen = records[best_idx]

    weights = chosen["weights"]
    lam = chosen["lambda"]

    # write configs
    (CONF_DIR / "aggregation.json").write_text(json.dumps({"half_life_days": float(best_hl)}, ensure_ascii=False, indent=2), encoding="utf-8")
    hosts = ds["host"].value_counts().index.tolist()
    sw_all = json.loads((CONF_DIR / "source_weights.json").read_text(encoding="utf-8")) if (CONF_DIR / "source_weights.json").exists() else {}
    for h in hosts:
        g = host_to_group(h, groups_map)
        sw_all[h] = weights.get(g, 1.0)
    (CONF_DIR / "source_weights.json").write_text(json.dumps(sw_all, ensure_ascii=False, indent=2), encoding="utf-8")

    # history & report
    hist_path = OUT_DIR / "source_weights_history.json"
    hist = []
    if hist_path.exists():
        try: hist = json.loads(hist_path.read_text(encoding="utf-8"))
        except Exception: hist = []
    hist.append({"ts": pd.Timestamp.now().isoformat(), "half_life": float(best_hl), "weights_group": weights, "cv_best_adj": float(chosen["mean"]), "lambda": lam})
    hist_path.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")

    rep = ["# Optimization summary",
           f"- Repeats×Folds: {repeats}×{len(ranges_all[0])} ; patience={int(CFG.get('patience',2))} ; rel_delta={float(CFG.get('early_rel_delta',0.01)):.2%}",
           f"- Best half-life (CV mean): {best_hl} days ≈ {best_val:.2f}",
           f"- λ by 1SE: λ={lam} ; threshold={threshold:.2f} ; best={best_mean:.2f}±{best_se:.2f}",
           "- Learned group weights:"]
    for k2,v in weights.items():
        rep.append(f"  - {k2}: {v:.2f}")
    (OUT_DIR / "optimize_report.md").write_text("\n".join(rep), encoding="utf-8")

if __name__ == "__main__":
    main()
