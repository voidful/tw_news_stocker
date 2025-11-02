# -*- coding: utf-8 -*-
# Sensitivity grid: half-life x TopN x min_score x risk_model -> Sharpe/Return/MaxDD ...
import json, pandas as pd, numpy as np
from pathlib import Path
from datetime import timedelta
import requests
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def compute_metric_weights(df_out, method="entropy", tau=1.0):
    # metrics: sharpe (higher), total_ret (higher), maxdd (smaller | we use -abs)
    import numpy as np, pandas as pd, math
    X = pd.DataFrame({
        "sharpe": df_out["sharpe"].astype(float),
        "total": df_out["total_ret"].astype(float),
        "maxdd_adj": -df_out["maxdd"].abs().astype(float)
    })
    # normalize per metric
    def minmax(s):
        lo, hi = float(s.min()), float(s.max())
        return (s - lo) / (hi - lo + 1e-12)
    Z = X.apply(minmax)

    if method == "entropy":
        eps = 1e-12
        P = Z.div(Z.sum(axis=0) + eps, axis=1).clip(lower=eps)  # probability-like per metric
        k = 1.0 / np.log(len(P)) if len(P) > 1 else 1.0
        E = -k * (P * np.log(P)).sum(axis=0)  # entropy per metric
        d = 1 - E
        w = d / (d.sum() + eps)
        return {"sharpe": float(w["sharpe"]), "maxdd": float(w["maxdd_adj"]), "total_ret": float(w["total"])}
    else:  # bma-like: evidence via log-sum-exp of standardized metric
        S = (Z - Z.mean())/(Z.std()+1e-12)
        evid = np.log(np.exp(S["sharpe"]/tau).sum()) , np.log(np.exp(S["maxdd_adj"]/tau).sum()) , np.log(np.exp(S["total"]/tau).sum())
        e = np.array(evid, dtype=float)
        e = np.exp(e - np.max(e))  # stabilize
        w = e / (e.sum() + 1e-12)
        return {"sharpe": float(w[0]), "maxdd": float(w[1]), "total_ret": float(w[2])}
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CONF_DIR = ROOT / "config"
NEWS_LOG = DATA_DIR / "news_log.jsonl"

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

def portfolio_stats(rets, freq=252):
    if len(rets)==0: return (0,0,0,0,0)
    r = pd.Series(rets)
    nav = (1+r).cumprod()
    total = nav.iloc[-1]-1
    vol = r.std()*(freq**0.5)
    sharpe = r.mean()/(r.std()+1e-12)*(freq**0.5)
    win = (r>0).mean()
    dd = (nav/nav.cummax()-1).min()
    return (float(sharpe), float(total), float(vol), float(win), float(dd))

def _weights_for(picks, prices, date, lookback=20, model="equal", vol_target=0.15, kelly_coef=0.5):
    if not picks: return {}
    sig = {}; mu = {}
    for c in picks:
        s = prices.get(c)
        if s is None or date not in s.index: 
            sig[c]=0.02; mu[c]=0.0; continue
        r = s.loc[:date].pct_change().dropna().tail(lookback)
        sig[c] = float(r.std() or 0.02)
        mu[c] = float(r.mean() or 0.0)
    if model=="invvol":
        inv = {c: 1.0/(sig[c]+1e-6) for c in picks}; z = sum(inv.values()) or 1.0
        w = {c: inv[c]/z for c in picks}
    elif model=="kelly":
        f = {c: max(0.0, mu[c]/((sig[c]**2)+1e-8))*kelly_coef for c in picks}; z = sum(f.values()) or 1.0
        w = {c: f[c]/z for c in picks}
    else:
        w = {c: 1.0/len(picks) for c in picks}
    # scale to vol target (diag cov)
    port_sigma = (sum((w[c]**2)*(sig[c]**2) for c in picks))**0.5
    target_daily = vol_target / (252**0.5)
    scale = min(1.5, target_daily/(port_sigma+1e-9)) if port_sigma>0 else 1.0
    return {c: w[c]*scale for c in picks}

def simple_daily_backtest(scores, prices, topN=5, min_score=1.0, risk_model="equal", vol_target=0.15, rebalance=1, kelly_coef=0.5):
    dates = pd.date_range(scores["ts"].min().date()+pd.Timedelta(days=1), scores["ts"].max().date(), freq="B")
    rets = []
    weights_prev = {}
    for i, d in enumerate(dates):
        cutoff = pd.Timestamp(d) - pd.Timedelta(days=1)
        sub = scores[scores["ts"]<=cutoff]
        if sub.empty: rets.append(0.0); continue
        rank = sub.groupby("code")["sig"].sum().sort_values(ascending=False)
        picks = rank[rank>=min_score].head(topN).index.tolist()
        if (i % rebalance)==0:
            weights = _weights_for(picks, prices, cutoff, model=risk_model, vol_target=vol_target, kelly_coef=kelly_coef)
        else:
            weights = weights_prev
        vals = []
        for c, w in (weights or {}).items():
            s = prices.get(c)
            if s is None or d not in s.index or (d+pd.Timedelta(days=1)) not in s.index:
                continue
            vals.append(w * float(s.loc[d+pd.Timedelta(days=1)]/s.loc[d]-1.0))
        rets.append(sum(vals) if vals else 0.0)
        weights_prev = weights
    return rets


def main():
    rows = []
    if not NEWS_LOG.exists(): raise SystemExit("no news")
    for line in NEWS_LOG.read_text(encoding="utf-8").splitlines():
        try: rows.append(json.loads(line))
        except: pass
    df = pd.DataFrame(rows); df["ts"]=pd.to_datetime(df["ts"])
    recs = []
    for _, r in df.iterrows():
        ts = r["ts"]; pc = r.get("per_company_raw") or r.get("per_company") or {}
        for code, val in (pc.items() if isinstance(pc, dict) else []):
            recs.append({"ts": ts, "code": code, "raw": float(val)})
    ds = pd.DataFrame(recs)
    if ds.empty: raise SystemExit("no signal")
    prices = load_prices(ds["code"].unique())

    CFG = json.loads((CONF_DIR / "optimize.json").read_text(encoding="utf-8")).read_text(encoding="utf-8"))
    hl_grid = CFG.get("hl_grid",[3,5,7,10,14,21,30])
    N_grid = CFG.get("grid_N",[3,5,10])
    VOL_grid = CFG.get("grid_vol_target",[0.10,0.15,0.20])
    REBAL_grid = CFG.get("grid_rebalance",[1,5])
    KELLY_grid = CFG.get("grid_kelly_coef",[0.25,0.5,0.75])
    M_grid = CFG.get("grid_min_score",[0.5,1.0,1.5])

    rows_out = []
for hl in hl_grid:
        tmp = ds.copy()
        tmp["age_days"] = (tmp["ts"].max() - tmp["ts"]).dt.total_seconds()/86400.0
        tmp["w"] = tmp["age_days"].apply(lambda x: time_decay_weight(x, hl))
        tmp["sig"] = tmp["raw"] * tmp["w"]
        for N in N_grid:
            for m in M_grid:
                for rm in risk_models:
                    for vt in V_grid:
                        for rb in Rb_grid:
                            for kc in Kc_grid:
                                rets = simple_daily_backtest(tmp, prices, topN=N, min_score=m, risk_model=rm, vol_target=vt, rebalance=rb, kelly_coef=kc)
                                s, tot, vol, win, dd = portfolio_stats(rets)
                                rows_out.append({"half_life": hl, "topN": N, "min_score": m, "risk_model": rm, "vol_target": vt, "rebalance_freq": rb, "kelly_coef": kc, "sharpe": s, "total_ret": tot, "vol": vol, "win": win, "maxdd": dd})
df_out = pd.DataFrame(rows_out)

    df_out.to_csv(OUT_DIR / "sensitivity_grid.csv", index=False, encoding="utf-8-sig")

    # Robustness summary
    piv = df_out.pivot_table(index=["half_life"], values=["sharpe","total_ret"], aggfunc=[np.median, np.mean, np.min, np.max])
    lines = ["# Robustness summary"]
    lines.append(piv.to_string())
    (OUT_DIR / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
