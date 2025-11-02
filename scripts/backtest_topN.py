# -*- coding: utf-8 -*-
# Advanced backtest with dynamic liquidity buckets, market regime, vol targeting, half-Kelly, and rebalance frequency.

import json, math
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CONF_DIR = ROOT / "config"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEWS_LOG = DATA_DIR / "news_log.jsonl"
AGG_CONF = json.loads((CONF_DIR / "aggregation.json").read_text(encoding="utf-8"))
RISK_CONF = json.loads((CONF_DIR / "risk.json").read_text(encoding="utf-8"))
HALF_LIFE = float(AGG_CONF.get("half_life_days", 7.0))

def time_decay_weight(age_days: float, half_life_days: float) -> float:
    return 2.0 ** (-age_days / half_life_days) if half_life_days > 0 else 1.0

def load_news():
    rows = []
    if NEWS_LOG.exists():
        for line in NEWS_LOG.read_text(encoding="utf-8").splitlines():
            try: rows.append(json.loads(line))
            except: pass
    if not rows:
        raise SystemExit("news_log.jsonl 為空，請先跑一次主流程。")
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df

def get_price_csv(code: str) -> pd.DataFrame:
    cache = DATA_DIR / "prices" / f"{code}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            return pd.read_csv(cache)
        except Exception:
            pass
    url = f"https://raw.githubusercontent.com/voidful/tw_stocker/main/data/{code}.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    cache.write_bytes(r.content)
    return pd.read_csv(cache)

class TwoRegimeHMM:
    def __init__(self, trans, mu, std):
        import numpy as np
        self.P = np.array(trans, dtype=float)  # 2x2
        self.mu = mu; self.std = std
        self.p = np.array([0.5, 0.5], dtype=float)

    def step(self, sigma_obs):
        import numpy as np, math
        # emission likelihoods ~ Normal(mu, std) on daily sigma
        def nlik(x,m,s): 
            s=max(s,1e-6); return math.exp(-0.5*((x-m)/s)**2)/ (s*(2*math.pi)**0.5)
        ell = np.array([nlik(sigma_obs, self.mu["low"], self.std["low"]),
                        nlik(sigma_obs, self.mu["high"], self.std["high"])], dtype=float)
        # prediction and update
        p_pred = self.P.T.dot(self.p)
        self.p = (ell * p_pred); self.p = self.p / (self.p.sum()+1e-12)
        return self.p

def load_market_index():

    # Try TWII from tw_stocker if available
    try:
        url = "https://raw.githubusercontent.com/voidful/tw_stocker/main/data/^TWII.csv"
        df = pd.read_csv(url)
        ts = pd.to_datetime(df["Datetime"] if "Datetime" in df.columns else df["date"])
        close = df.get("close", df.get("Close", df.get("c")))
        s = pd.Series(close.values, index=ts).resample("1D").last().dropna()
        return s
    except Exception:
        return None

def daily_series(df: pd.DataFrame):
    # Return (close, volume) daily series
    if "Datetime" in df.columns:
        ts = pd.to_datetime(df["Datetime"]); base = df
    elif "date" in df.columns:
        ts = pd.to_datetime(df["date"]); base = df
    else:
        raise ValueError("無 Datetime/date 欄位")
    # price
    for c in ["close","Close","c"]:
        if c in base.columns:
            close = base[c]; break
    else:
        raise ValueError("找不到 close 欄位")
    s_close = pd.Series(close.values, index=ts).resample("1D").last().dropna()
    # volume
    vol = None
    for v in ["volume","Volume","成交量","vol"]:
        if v in base.columns:
            vol = base[v]; break
    if vol is not None:
        s_vol = pd.Series(vol.values, index=ts).resample("1D").sum().dropna()
    else:
        s_vol = None
    return s_close, s_vol

def portfolio_stats(returns: pd.Series, freq=252):
    if returns.empty:
        return {"CAGR":0, "Vol":0, "Sharpe":0, "WinRate":0, "MaxDD":0, "TotalRet":0}
    cum = (1+returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    years = len(returns)/freq
    cagr = (1+total_ret)**(1/max(years,1e-9)) - 1
    vol = returns.std() * (freq**0.5)
    sharpe = returns.mean()/ (returns.std()+1e-12) * (freq**0.5)
    peak = cum.cummax()
    dd = (cum/peak - 1).min()
    win = (returns>0).mean()
    return {"CAGR":cagr, "Vol":vol, "Sharpe":sharpe, "WinRate":win, "MaxDD":dd, "TotalRet":total_ret}

def calibrate_piecewise(code, price_series, volume_series, lookback=60, segs=None):
    # Estimate per-name segment multipliers from illiquidity proxy over lookback days
    # segs: [{"p":0.01,"mult":...}, ...]; we fit multipliers by matching bins of Amihud proxy percentiles
    import numpy as np
    p = price_series.get(code); v = volume_series.get(code)
    if p is None or v is None or len(p)<lookback+5 or len(v)<lookback+5:
        return None
    r = p.pct_change().dropna().tail(lookback)
    val = (p*v).dropna().reindex(r.index).fillna(method="ffill")
    if val.isna().all(): return None
    illiq = (r.abs()/(val.replace(0, np.nan))).dropna()  # Amihud
    if illiq.empty: return None
    # bin by participation proxies (use seg thresholds p)
    segs = segs or [{"p":0.01},{"p":0.03},{"p":0.05},{"p":0.10}]
    qs = np.quantile(illiq, [min(0.99, s["p"]*10) for s in segs])  # crude mapping: higher p -> higher illiquidity
    # map each seg to multiplier proportional to avg illiquidity in bin
    mults = []
    prev = 0.0
    vals = illiq.values
    for i, s in enumerate(segs):
        thr = qs[i]
        bin_vals = vals[(vals>=prev) & (vals<=thr)]
        m = float(np.mean(bin_vals)) if len(bin_vals)>0 else float(np.mean(vals))
        mults.append(m)
        prev = thr
    # normalize around 1.0
    base = float(np.mean(mults)) if mults else 1.0
    return [float(x/(base+1e-12)) for x in mults]

calib_cache = {}
calib_last = {}

def micro_cost_model(weights, prev_weights, prices, vols, adv, vol_20, gamma=0.5):
(weights, prev_weights, price_series, volume_series, adv, vol_20, gamma=0.5):
    # Square-root impact on turnover; fall back handled by caller.
    tw = sum(abs(weights.get(k,0)-prev_weights.get(k,0)) for k in set(list(weights.keys())+list(prev_weights.keys())))
    if adv is None or vol_20 is None:
        return 0.0, tw
    q = tw  # portfolio turnover fraction
    sigma = np.nanmean(list(vol_20.values())) if vol_20 else 0.02
    impact = gamma * sigma * (q ** 0.5)
    return float(impact), tw

def weight_scheme(picks, price_series, lookback=20, model="equal", vol_target=0.15, kelly_cap=0.25, kelly_coef=0.5):
    if not picks:
        return {}, 0.0
    # estimate individual daily vol and mu
    sig = {}
    mu = {}
    for c in picks:
        s = price_series.get(c)
        if s is None or len(s) < lookback+2:
            sig[c] = np.nan; mu[c] = 0.0; continue
        r = s.pct_change().dropna()
        sig[c] = float(r.tail(lookback).std())
        mu[c] = float(r.tail(lookback).mean())
    if model == "invvol":
        inv = {c: (1.0/(sig[c]+1e-6)) for c in picks}
        ssum = sum(inv.values()) or 1.0
        w = {c: inv[c]/ssum for c in picks}
    elif model == "kelly":
        f = {c: max(0.0, mu[c]/((sig[c]**2)+1e-8)) * kelly_coef for c in picks}
        f = {c: min(kelly_cap, v) for c,v in f.items()}
        ssum = sum(f.values()) or 1.0
        w = {c: f[c]/ssum for c in picks}
    elif model == "semivar":
        inv = {}
        for c in picks:
            s = price_series.get(c)
            if s is None:
                inv[c]=0.0; continue
            r = s.pct_change().dropna()
            d = r[r<0].tail(lookback)
            sv = float(np.sqrt((d**2).mean()+1e-12))
            inv[c] = 1.0/sv if sv>0 else 0.0
        ssum = sum(inv.values()) or 1.0
        w = {c: inv[c]/ssum for c in picks}
    else:
        w = {c: 1.0/len(picks) for c in picks}
    # vol targeting (independence approx)
    port_sigma_est = math.sqrt(sum((w[c]**2) * ((sig[c] or 0.02)**2) for c in picks))
    target_daily = vol_target / (252**0.5)
    scale = min(1.5, max(0.0, target_daily / (port_sigma_est + 1e-9))) if port_sigma_est>0 else 1.0
    w = {c: v*scale for c,v in w.items()}
    return w, port_sigma_est*scale

def run_backtest(start_date=None, end_date=None, N=None, min_score=None, market_index_code="0050"):
    df = load_news()
    if start_date is None:
        start_date = (df["ts"].min() + pd.Timedelta(days=7)).date()
    if end_date is None:
        end_date = df["ts"].max().date()
    if N is None: N = int(RISK_CONF.get("max_positions", 5))
    if min_score is None: min_score = float(RISK_CONF.get("min_score", 1.0))

    dates = pd.date_range(start_date, end_date, freq="B")
    # load price & volume
    price_series = {}
    volume_series = {}
    all_codes = sorted({c for codes in df["codes"] for c in (codes or [])})
    for code in all_codes:
        try:
            close_df = get_price_csv(code)
            sc, sv = daily_series(close_df)
            price_series[code] = sc
            volume_series[code] = sv
        except Exception:
            price_series[code] = None
            volume_series[code] = None
    # market index
    try:
        idx_df = get_price_csv(market_index_code)
        index_series, _ = daily_series(idx_df)
    except Exception:
        index_series = None

    holding_days = int(RISK_CONF.get("holding_days", 3))
    dd_stop = float(RISK_CONF.get("dd_stop", 0.2))
    cooloff_days = int(RISK_CONF.get("cooloff_days", 10))
    risk_model = str(RISK_CONF.get("risk_model", "invvol"))
    vol_target = float(RISK_CONF.get("vol_target", 0.15))
    lookback = int(RISK_CONF.get("lookback_days", 20))
    fallback_tc_bps = float(RISK_CONF.get("tc_bps", 10.0))
    fallback_slip_bps = float(RISK_CONF.get("slippage_bps", 5.0))
    rebalance = int(RISK_CONF.get("rebalance_freq_days", 1)) or 1
    kelly_cap = float(RISK_CONF.get("kelly_cap", 0.25))
    kelly_coef = float(RISK_CONF.get("kelly_coef", 0.5))

    # signals
    sig_rows = []
    for _, r in df.iterrows():
        ts = r["ts"]; pc = r.get("per_company") or {}
        for code, val in pc.items():
            sig_rows.append({"ts": ts, "code": code, "raw": float(val)})
    sdf = pd.DataFrame(sig_rows)
    if sdf.empty:
        raise SystemExit("無 per_company 欄位；請先跑新版主流程。")

    idx_series = load_market_index()
    # init HMM
    hmm = TwoRegimeHMM(trans=RISK_CONF.get("hmm_trans", [[0.95,0.05],[0.05,0.95]]), mu=RISK_CONF.get("hmm_sigma_mu", {"low":0.012,"high":0.028}), std=RISK_CONF.get("hmm_sigma_std", {"low":0.006,"high":0.010}))
    nav = 1.0
    peak = 1.0
    in_cooloff = 0
    weights_prev = {}  # code -> weight

    records = []

    for i, d in enumerate(dates):
        d0 = pd.Timestamp(d).normalize()
        cutoff = d0 - pd.Timedelta(days=1)
        sub = sdf[sdf["ts"] <= cutoff]
        if not sub.empty:
            sub = sub.copy()
            sub["age_days"] = (cutoff - sub["ts"]).dt.total_seconds()/86400.0
            sub["w"] = sub["age_days"].apply(lambda x: time_decay_weight(x, HALF_LIFE))
            sub["sig"] = sub["raw"] * sub["w"]
            rank = sub.groupby("code")["sig"].sum().sort_values(ascending=False)
            picks = [c for c in rank.index.tolist() if rank.loc[c] >= min_score][:N]
        else:
            picks = []

        if in_cooloff > 0 or not picks:
            target_weights = {}
        else:
            if i % rebalance == 0:
                target_weights, _ = weight_scheme(picks, price_series, lookback=lookback, model=risk_model, vol_target=vol_target, kelly_cap=kelly_cap, kelly_coef=kelly_coef)
            else:
                target_weights = weights_prev

        # gross return
        gross = 0.0
        if target_weights:
            vals = []
            for c, w in target_weights.items():
                s = price_series.get(c)
                if s is None or d0 not in s.index or (d0+pd.Timedelta(days=1)) not in s.index:
                    continue
                r = float(s.loc[d0+pd.Timedelta(days=1)]/s.loc[d0] - 1.0)
                vals.append(w * r)
            gross = sum(vals)

        # microstructure costs with dynamic buckets and market regime
        adv = {}; vol20 = {}; adv_vals = []
        for c in picks:
            p = price_series.get(c); v = volume_series.get(c)
            if p is not None and v is not None and d0 in p.index and d0 in v.index:
                pp = p.loc[:d0].tail(lookback)
                vv = v.loc[:d0].tail(lookback)
                if len(pp)>5 and len(vv)>5:
                    a = float((pp*vv).mean()); adv[c]=a; adv_vals.append(a)
                    rc = pp.pct_change().dropna()
                    vol20[c] = float(rc.std())
                else:
                    adv[c]=None; vol20[c]=None
            else:
                adv[c]=None; vol20[c]=None

        liq_q = RISK_CONF.get("liquidity_quantiles", [0.2,0.5,0.8])
        liq_mult = RISK_CONF.get("liquidity_multipliers", [1.4,1.1,0.9,0.7])
        buckets = {}
        if adv_vals:
            qs = [np.quantile(adv_vals, q) for q in liq_q]
            for c,a in adv.items():
                if a is None: buckets[c]=0
                elif a<=qs[0]: buckets[c]=0
                elif a<=qs[1]: buckets[c]=1
                elif a<=qs[2]: buckets[c]=2
                else: buckets[c]=3
        else:
            buckets = {c:1 for c in picks}

        gamma_low = float(RISK_CONF.get("regime_gamma", {}).get("low", 0.4))
        gamma_high = float(RISK_CONF.get("regime_gamma", {}).get("high", 0.7))
        thr = float(RISK_CONF.get("regime_threshold_daily_sigma", 0.02))
        gamma = gamma_low
        if index_series is not None and d0 in index_series.index:
            rc = index_series.loc[:d0].tail(lookback).pct_change().dropna()
            if len(rc)>5 and float(rc.std()) > thr:
                gamma = gamma_high

        micro_cost, turnover = micro_cost_model(target_weights, weights_prev, price_series, volume_series, adv, vol20, gamma=gamma)
        if target_weights and liq_mult:
            avg_mult = np.mean([liq_mult[buckets.get(c,1)] for c in target_weights.keys()])
            micro_cost *= float(avg_mult)

        if micro_cost == 0.0 and target_weights:
            prev_set = set(weights_prev.keys()); next_set = set(target_weights.keys())
            sells = prev_set - next_set; buys = next_set - prev_set
            turnover_frac = (len(sells)+len(buys))/max(N,1)
            micro_cost = turnover_frac * (fallback_tc_bps/10000.0) + (len(buys)/max(N,1))*(fallback_slip_bps/10000.0)

        net = gross - micro_cost
        nav = nav * (1 + net)
        peak = max(peak, nav)
        dd = (nav/peak - 1.0)
        if dd <= -dd_stop and in_cooloff == 0:
            in_cooloff = cooloff_days
        elif in_cooloff > 0:
            in_cooloff -= 1

        records.append({"date": d0, "ret": net, "nav": nav, "dd": dd, "turnover": turnover, "holds": ",".join(sorted(target_weights.keys()))})
        weights_prev = target_weights

    df_out = pd.DataFrame(records).sort_values("date")
    stats = portfolio_stats(df_out["ret"].fillna(0.0))
    df_out.to_csv(OUT_DIR / "backtest_enhanced.csv", index=False, encoding="utf-8-sig")

    lines = ["# Enhanced Backtest Report",
             f"- Half-life: {HALF_LIFE} days",
             f"- Risk model: {risk_model}, vol_target={vol_target}, rebalance={rebalance}d, kelly_coef={kelly_coef}",
             f"- Max positions: {N}, Min score: {min_score}",
             f"- Costs: microstructure sqrt-impact with dynamic liquidity buckets; regime gamma based on index sigma",
             "",
             "## Performance"]
    lines += [f"- 累積報酬：{stats['TotalRet']*100:.2f}%", f"- 年化報酬：{stats['CAGR']*100:.2f}%",
              f"- 年化波動：{stats['Vol']*100:.2f}%", f"- 夏普比：{stats['Sharpe']:.2f}",
              f"- 勝率：{stats['WinRate']*100:.2f}%", f"- 最大回撤：{stats['MaxDD']*100:.2f}%"]
    (OUT_DIR / "backtest_enhanced_report.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    cfg = json.loads((CONF_DIR / "backtest_config.json").read_text(encoding="utf-8")) if (CONF_DIR / "backtest_config.json").exists() else {}
    run_backtest(cfg.get("start_date"), cfg.get("end_date"), cfg.get("N_list", [5])[0] if isinstance(cfg.get("N_list"), list) else cfg.get("N_list"), cfg.get("min_score"))
