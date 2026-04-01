# -*- coding: utf-8 -*-
# Sensitivity grid: half-life x TopN x min_score x risk_model -> Sharpe/Return/MaxDD ...
import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CONF_DIR = ROOT / "config"
NEWS_LOG = DATA_DIR / "news_log.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metric_weights(df_out, method="entropy", tau=1.0):
    """Return weights for (sharpe, maxdd_adj, total_ret)."""
    X = pd.DataFrame({
        "sharpe": df_out["sharpe"].astype(float),
        "total": df_out["total_ret"].astype(float),
        "maxdd_adj": -df_out["maxdd"].abs().astype(float),
    })

    def minmax(s: pd.Series) -> pd.Series:
        lo, hi = float(s.min()), float(s.max())
        return (s - lo) / (hi - lo + 1e-12)

    Z = X.apply(minmax)

    if method == "entropy":
        eps = 1e-12
        P = Z.div(Z.sum(axis=0) + eps, axis=1).clip(lower=eps)
        k = 1.0 / np.log(len(P)) if len(P) > 1 else 1.0
        E = -k * (P * np.log(P)).sum(axis=0)            # 熵
        d = 1 - E                                       # 差異度
        w = d / (d.sum() + eps)
        return {"sharpe": float(w["sharpe"]), "maxdd": float(w["maxdd_adj"]), "total_ret": float(w["total"])}
    else:
        # BMA-like: log-sum-exp 的證據聚合
        S = (Z - Z.mean()) / (Z.std() + 1e-12)
        evid = (
            np.log(np.exp(S["sharpe"] / tau).sum()),
            np.log(np.exp(S["maxdd_adj"] / tau).sum()),
            np.log(np.exp(S["total"] / tau).sum()),
        )
        e = np.array(evid, dtype=float)
        e = np.exp(e - np.max(e))                       # 數值穩定化
        w = e / (e.sum() + 1e-12)
        return {"sharpe": float(w[0]), "maxdd": float(w[1]), "total_ret": float(w[2])}


def load_prices(codes):
    """讀取個股日收盤價，回傳 {code: pd.Series}（日頻）。"""
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
            # 忽略缺資料的代號
            continue
    return prices


def time_decay_weight(age_days: float, half_life_days: float) -> float:
    return 2.0 ** (-age_days / half_life_days) if half_life_days > 0 else 1.0


def portfolio_stats(rets, freq=252):
    """回傳 (sharpe, total_ret, vol, win_rate, maxdd)。"""
    if len(rets) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    r = pd.Series(rets)
    nav = (1 + r).cumprod()
    total = float(nav.iloc[-1] - 1.0)
    vol = float(r.std() * (freq ** 0.5))
    sharpe = float(r.mean() / (r.std() + 1e-12) * (freq ** 0.5))
    win = float((r > 0).mean())
    dd = float((nav / nav.cummax() - 1).min())
    return (sharpe, total, vol, win, dd)


def _weights_for(picks, prices, date, lookback=20, model="equal", vol_target=0.15, kelly_coef=0.5):
    if not picks:
        return {}
    sig = {}
    mu = {}
    for c in picks:
        s = prices.get(c)
        if s is None or date not in s.index:
            sig[c] = 0.02
            mu[c] = 0.0
            continue
        r = s.loc[:date].pct_change().dropna().tail(lookback)
        sig[c] = float(r.std() or 0.02)
        mu[c] = float(r.mean() or 0.0)

    if model == "invvol":
        inv = {c: 1.0 / (sig[c] + 1e-6) for c in picks}
        z = sum(inv.values()) or 1.0
        w = {c: inv[c] / z for c in picks}
    elif model == "kelly":
        f = {c: max(0.0, mu[c] / ((sig[c] ** 2) + 1e-8)) * kelly_coef for c in picks}
        z = sum(f.values()) or 1.0
        w = {c: f[c] / z for c in picks}
    else:
        w = {c: 1.0 / len(picks) for c in picks}

    # 目標波動縮放（對角近似）
    port_sigma = (sum((w[c] ** 2) * (sig[c] ** 2) for c in picks)) ** 0.5
    target_daily = vol_target / (252 ** 0.5)
    scale = min(1.5, target_daily / (port_sigma + 1e-9)) if port_sigma > 0 else 1.0
    return {c: w[c] * scale for c in picks}


def simple_daily_backtest(scores, prices, topN=5, min_score=1.0, risk_model="equal", vol_target=0.15, rebalance=1, kelly_coef=0.5):
    """以訊號日後一天起算，模擬等頻再平衡日報酬。"""
    dates = pd.date_range(scores["ts"].min().date() + pd.Timedelta(days=1),
                          scores["ts"].max().date(), freq="B")
    rets = []
    weights_prev = {}
    for i, d in enumerate(dates):
        cutoff = pd.Timestamp(d) - pd.Timedelta(days=1)
        sub = scores[scores["ts"] <= cutoff]
        if sub.empty:
            rets.append(0.0)
            continue

        rank = sub.groupby("code")["sig"].sum().sort_values(ascending=False)
        picks = rank[rank >= min_score].head(topN).index.tolist()

        if (i % rebalance) == 0:
            weights = _weights_for(picks, prices, cutoff, model=risk_model,
                                   vol_target=vol_target, kelly_coef=kelly_coef)
        else:
            weights = weights_prev

        vals = []
        for c, w in (weights or {}).items():
            s = prices.get(c)
            # 用收盤到下一個交易日收盤的報酬
            if s is None or d not in s.index or (d + pd.Timedelta(days=1)) not in s.index:
                continue
            vals.append(w * float(s.loc[d + pd.Timedelta(days=1)] / s.loc[d] - 1.0))

        rets.append(sum(vals) if vals else 0.0)
        weights_prev = weights
    return rets


def main():
    # 讀取新聞逐筆原始分數
    if not NEWS_LOG.exists():
        raise SystemExit("no news")

    rows = []
    for line in NEWS_LOG.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no news")

    df["ts"] = pd.to_datetime(df["ts"])

    recs = []
    for _, r in df.iterrows():
        ts = r["ts"]
        pc = r.get("per_company_raw") or r.get("per_company") or {}
        if isinstance(pc, dict):
            for code, val in pc.items():
                recs.append({"ts": ts, "code": code, "raw": float(val)})

    ds = pd.DataFrame(recs)
    if ds.empty:
        raise SystemExit("no signal")

    # 價格資料
    prices = load_prices(ds["code"].unique())

    # 讀取網格設定
    try:
        cfg_text = (CONF_DIR / "optimize.json").read_text(encoding="utf-8")
        CFG = json.loads(cfg_text)
    except Exception:
        CFG = {}

    hl_grid     = CFG.get("hl_grid",           [3, 5, 7, 10, 14, 21, 30])
    N_grid      = CFG.get("grid_N",            [3, 5, 10])
    VOL_grid    = CFG.get("grid_vol_target",   [0.10, 0.15, 0.20])
    REBAL_grid  = CFG.get("grid_rebalance",    [1, 5])
    KELLY_grid  = CFG.get("grid_kelly_coef",   [0.25, 0.5, 0.75])
    M_grid      = CFG.get("grid_min_score",    [0.5, 1.0, 1.5])
    RISK_grid   = CFG.get("grid_risk_model",   ["equal", "invvol", "kelly"])

    rows_out = []
    for hl in hl_grid:
        tmp = ds.copy()
        tmp["age_days"] = (tmp["ts"].max() - tmp["ts"]).dt.total_seconds() / 86400.0
        tmp["w"] = tmp["age_days"].apply(lambda x: time_decay_weight(x, hl))
        tmp["sig"] = tmp["raw"] * tmp["w"]

        for N in N_grid:
            for m in M_grid:
                for rm in RISK_grid:
                    for vt in VOL_grid:
                        for rb in REBAL_grid:
                            for kc in KELLY_grid:
                                rets = simple_daily_backtest(
                                    tmp, prices,
                                    topN=N, min_score=m,
                                    risk_model=rm, vol_target=vt,
                                    rebalance=rb, kelly_coef=kc
                                )
                                s, tot, vol, win, dd = portfolio_stats(rets)
                                rows_out.append({
                                    "half_life": hl,
                                    "topN": N,
                                    "min_score": m,
                                    "risk_model": rm,
                                    "vol_target": vt,
                                    "rebalance_freq": rb,
                                    "kelly_coef": kc,
                                    "sharpe": s,
                                    "total_ret": tot,
                                    "vol": vol,
                                    "win": win,
                                    "maxdd": dd
                                })

    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(OUT_DIR / "sensitivity_grid.csv", index=False, encoding="utf-8-sig")

    # Robust 加權排名（entropy/bma）
    try:
        method = CFG.get("robust_weight_method", "entropy")
        tau = float(CFG.get("bma_tau", 1.0))
        # 權重
        RW = compute_metric_weights(df_out, method=method, tau=tau)
        # 依權重計算 robust_score 與平均名次
        Z = pd.DataFrame({
            "sharpe": df_out["sharpe"].astype(float),
            "total_ret": df_out["total_ret"].astype(float),
            "maxdd_adj": -df_out["maxdd"].abs().astype(float),
        })
        # min-max 正規化
        Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-12)
        robust_score = (
            Z["sharpe"] * RW["sharpe"] +
            Z["total_ret"] * RW["total_ret"] +
            Z["maxdd_adj"] * RW["maxdd"]
        )
        rank_sharpe = df_out["sharpe"].rank(ascending=False)
        rank_total  = df_out["total_ret"].rank(ascending=False)
        rank_dd     = (-df_out["maxdd"].abs()).rank(ascending=False)
        avg_rank = (rank_sharpe + rank_total + rank_dd) / 3.0

        df_rank = df_out.copy()
        df_rank["robust_score"] = robust_score
        df_rank["avg_rank"] = avg_rank
        df_rank.sort_values(["robust_score", "avg_rank"], ascending=[False, True], inplace=True)
        df_rank.to_csv(OUT_DIR / "sensitivity_rank.csv", index=False, encoding="utf-8-sig")
    except Exception:
        # 若計分失敗，略過但保留主輸出
        pass

    # Summary
    piv = df_out.pivot_table(index=["half_life"],
                             values=["sharpe", "total_ret"],
                             aggfunc=[np.median, np.mean, np.min, np.max])
    lines = ["# Robustness summary", piv.to_string()]
    (OUT_DIR / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()