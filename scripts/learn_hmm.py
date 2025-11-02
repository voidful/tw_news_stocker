# -*- coding: utf-8 -*-
# Learn K-state HMM on market daily volatility via EM (multi-start) with AIC/BIC model selection.
# Optional Student-t emission (nu fixed) for robustness.
import pandas as pd, numpy as np, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CONF = ROOT / "config"
OUT.mkdir(exist_ok=True, parents=True)

def load_index():
    try:
        url = "https://raw.githubusercontent.com/voidful/tw_stocker/main/data/^TWII.csv"
        df = pd.read_csv(url)
        ts = pd.to_datetime(df["Datetime"] if "Datetime" in df.columns else df["date"])
        close = df.get("close", df.get("Close", df.get("c")))
        s = pd.Series(close.values, index=ts).resample("1D").last().dropna()
        r = s.pct_change().dropna()
        sig = r.rolling(20).std().dropna()
        return sig
    except Exception:
        return None

def digamma(x):
    import math
    r = 0.0
    while x < 7:
        r -= 1/x
        x += 1
    f = 1.0/(x*x)
    return r + math.log(x) - 0.5/x - f*(1.0/12 - f*(1.0/120 - f*(1.0/252)))

def trigamma(x):
    import math
    r = 0.0
    while x < 7:
        r += 1/(x*x)
        x += 1
    f = 1.0/(x*x)
    return r + 0.5*f + f/x + f*f*(1.0/6 - f*(1.0/30))

def newton_update_nu(nu_init, log_tau_mean, tau_mean, iters=20, tol=1e-6):
    import math
    nu = max(2.01, float(nu_init))
    for _ in range(iters):
        f = math.log(nu/2.0) - digamma(nu/2.0) + 1.0 - (log_tau_mean - tau_mean)
        fp = 1.0/nu - 0.5*trigamma(nu/2.0)
        nu_new = max(2.001, nu - f/fp)
        if abs(nu_new - nu) < tol:
            nu = nu_new; break
        nu = nu_new
    return float(nu)

def t_pdf(x, mu, sigma, nu):
    # Student-t density
    sigma = max(float(sigma), 1e-6)
    nu = max(float(nu), 2.0)
    c = math.gamma((nu+1)/2)/ (math.sqrt(nu*math.pi)*math.gamma(nu/2))
    z = (x-mu)/sigma
    return c * (1 + (z*z)/nu) ** (-(nu+1)/2) / sigma

def em_hmm(obs, K=2, seeds=(0,1,2), max_iter=100, tol=1e-5, emission="gauss", nu=5.0, learn_nu=False):
    x = obs.values.astype(float); n = len(x)
    best = None
    for seed in seeds:
        np.random.seed(seed)
        # init
        qs = np.linspace(10, 90, K+2)[1:-1]
        centers = np.percentile(x, qs)
        order = np.argsort(centers)
        mu = centers[order].astype(float)
        std = np.clip(np.full(K, np.std(x)), 1e-4, None)
        A = np.full((K,K), 1.0/K); A += np.eye(K); A = A / A.sum(axis=1, keepdims=True)
        pi = np.full(K, 1.0/K)
        ll_prev = -1e18

        for _ in range(max_iter):
            # E-step
            if emission == "t":
                B = np.stack([t_pdf(x, mu[k], std[k], nu) for k in range(K)], axis=1)  # n x K
            else:
                # Gaussian
                def gpdf(xx,m,s):
                    s=max(s,1e-6)
                    return np.exp(-0.5*((xx-m)/s)**2)/(s*np.sqrt(2*np.pi))
                B = np.stack([gpdf(x, mu[k], std[k]) for k in range(K)], axis=1)
            B = np.clip(B, 1e-300, None)
            alpha = np.zeros((n,K)); beta = np.zeros((n,K)); c = np.zeros(n)
            alpha[0] = pi * B[0]; c[0] = alpha[0].sum()+1e-18; alpha[0]/=c[0]
            for t in range(1,n):
                alpha[t] = (alpha[t-1] @ A) * B[t]
                c[t] = alpha[t].sum()+1e-18; alpha[t]/=c[0 if t==0 else t]
            beta[-1] = np.ones(K)/c[-1]
            for t in range(n-2,-1,-1):
                beta[t] = (A @ (B[t+1]*beta[t+1]))/c[t]
            gamma = alpha*beta; gamma /= gamma.sum(axis=1, keepdims=True)+1e-18
            xi = np.zeros((n-1,K,K))
            for t in range(n-1):
                num = (alpha[t][:,None] * A) * (B[t+1][None,:] * beta[t+1][None,:])
                den = num.sum()+1e-18
                xi[t] = num/den

            # M-step
            pi = gamma[0]
            A = xi.sum(axis=0); A = A/ (A.sum(axis=1, keepdims=True)+1e-18)
            # emissions
            for k in range(K):
                w = gamma[:,k]
                wsum = w.sum()+1e-18
                if emission == "t":
                    z = (x - mu[k]) / (std[k]+1e-9)
                    wk = (nu+1.0) / (nu + z*z)
                    w_eff = w * wk
                    mu[k] = (w_eff * x).sum() / (w_eff.sum()+1e-18)
                    var = (w_eff * (x - mu[k])**2).sum() / (w_eff.sum()+1e-18)
                    std[k] = max(np.sqrt(var), 1e-6)
                else:
                    mu[k] = (w * x).sum() / wsum
                    var = (w * (x - mu[k])**2).sum() / wsum
                    std[k] = max(np.sqrt(var), 1e-6)

            if emission == "t" and learn_nu:
            Zlog = 0.0; Ztau = 0.0; denom = gamma.sum()+1e-18
            for k2 in range(K):
                z2 = ((x - mu[k2])/(std[k2]+1e-9))**2
                tau = (nu+1.0)/(nu + z2)
                Zlog += (gamma[:,k2]*np.log(tau+1e-18)).sum()
                Ztau += (gamma[:,k2]*tau).sum()
            log_tau_mean = Zlog/denom; tau_mean = Ztau/denom
            nu = newton_update_nu(nu, log_tau_mean, tau_mean)
        ll = np.sum(np.log(c+1e-18))
            if abs(ll-ll_prev) < tol:
                break
            ll_prev = ll

        params = {"trans": A.tolist(),
                  "mu": [float(m) for m in mu.tolist()],
                  "std": [float(s) for s in std.tolist()],
                  "pi": [float(p) for p in pi.tolist()],
                  "loglik": float(ll_prev)}
        if (best is None) or (params["loglik"] > best["loglik"]):
            best = params
    # sort states by mu ascending
    idx = np.argsort(best["mu"])
    A = np.array(best["trans"])[idx][:,idx]
    mu = np.array(best["mu"])[idx]
    std = np.array(best["std"])[idx]
    pi = np.array(best["pi"])[idx]
    return {"K": int(len(mu)),
            "trans": A.tolist(),
            "mu": {"low": float(mu[0]), "high": float(mu[-1])} if len(mu)>=2 else {"low": float(mu[0]), "high": float(mu[0])},
            "std": {"low": float(std[0]), "high": float(std[-1])},
            "pi": pi.tolist(),
            "loglik": float(best["loglik"])}

def model_select(obs, Ks=(2,3,4), seeds=(0,1,2,3), criterion="bic", emission="gauss", nu=5.0):
    n = len(obs)
    best = None
    cand = []
    for K in Ks:
        res = em_hmm(obs, K=K, seeds=seeds, emission=emission, nu=nu)
        k_params = K*(K-1) + K-1 + 2*K  # A (rows sum=1), pi (sum=1), mu/std per state
        if emission=="t": k_params += 0  # nu fixed
        aic = 2*k_params - 2*res["loglik"]
        bic = k_params*np.log(n) - 2*res["loglik"]
        res["AIC"] = float(aic); res["BIC"] = float(bic)
        cand.append(res)
    key = "BIC" if criterion.lower()=="bic" else "AIC"
    best = sorted(cand, key=lambda r: r[key])[0]
    return best, cand

def main():
    s = load_index()
    if s is None or s.empty:
        raise SystemExit("index not available")
    try:
        cfg_risk = json.loads((CONF / "risk.json").read_text(encoding="utf-8"))
    except Exception:
        cfg_risk = {}
    criterion = cfg_risk.get("hmm_select", "bic")
    Ks = cfg_risk.get("hmm_states", [2,3,4])
    emission = cfg_risk.get("hmm_emission", "gauss")
    nu0 = float(cfg_risk.get("hmm_t_nu", 5.0))
    learn_nu = bool(cfg_risk.get("hmm_learn_nu", False))
    use_bma = bool(cfg_risk.get("hmm_use_bma", True))

    obs = s.dropna()
    if use_bma:
        ens = model_ensemble_bma(obs, Ks=Ks, seeds=(0,1,2,3,4), emission=emission, nu=nu0, learn_nu=learn_nu)
        (CONF / "hmm_ensemble.json").write_text(json.dumps(ens, ensure_ascii=False, indent=2), encoding="utf-8")
        # best by chosen criterion also saved
        best, cand = model_select(obs, Ks=Ks, seeds=(0,1,2,3,4), criterion=criterion, emission=emission, nu=nu0)
        (CONF / "hmm_learned.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT / "hmm_report.md").write_text("
".join([
            "# HMM BMA ensemble",
            f"emission={emission}, Ks={Ks}, learn_nu={learn_nu}",
            "top weights: " + ", ".join([f"K={c['K']} w={c['weight']:.2f}" for c in ens[:5]])
        ]), encoding="utf-8")
    else:
        best, cand = model_select(obs, Ks=Ks, seeds=(0,1,2,3,4), criterion=criterion, emission=emission, nu=nu0)
        (CONF / "hmm_learned.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT / "hmm_report.md").write_text("
".join([
            "# HMM learned from market volatility",
            f"criterion={criterion}, emission={emission}, Ks={Ks}",
            f"selected: K={best['K']}",
            f"trans={best['trans']}",
            f"mu={best['mu']}", f"std={best['std']}",
            f"loglik={best['loglik']:.2f}"
        ]), encoding="utf-8")

if __name__ == "__main__":
    main()
def model_ensemble_bma(obs, Ks=(2,3,4), seeds=(0,1,2,3), emission="gauss", nu=5.0, learn_nu=False):
    cand = []
    n = len(obs)
    for K in Ks:
        for sd in seeds:
            res = em_hmm(obs, K=K, seeds=(sd,), emission=emission, nu=nu, learn_nu=learn_nu)
            k_params = K*(K-1) + K-1 + 2*K + (1 if (emission=="t" and learn_nu) else 0)
            bic = k_params*np.log(n) - 2*res["loglik"]
            cand.append({"K": K, "params": res, "bic": float(bic), "loglik": float(res["loglik"])})
    import numpy as np
    evid = np.array([-0.5 * c["bic"] for c in cand], dtype=float)
    w = np.exp(evid - np.max(evid)); w = w / (w.sum()+1e-12)
    for i,c in enumerate(cand): c["weight"] = float(w[i])
    return cand
