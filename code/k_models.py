# E10 blind-prediction contest: schedule-to-loss models fitted on the SAME constant/step/pulse runs, then asked to predict the
# Kovacs runs (never seen in fitting). Each model maps a full learning-rate history eta_0..eta_t (from initialisation, warmup
# included) to held-out loss; predicted excursions are L(Kovacs history) - L(reference history) at the evaluated steps.
#   tissue  single-timescale law (Tissue et al. 2024/NeurIPS 2025): L0 + A S1^-alpha - C S2, S2 = sum of momentum-filtered decrements
#   mpl     multi-power law (Luo et al. ICLR 2025): L0 + A (S1+S_W)^-alpha - B sum_k (eta_{k-1}-eta_k) G(eta_k^-gamma S_k(t)),
#           G(x) = 1 - (C x + 1)^-beta
#   nqm2    river R(S1) + two relaxing modes with free rates (rate ~ eta, equilibrium excess ~ eta)
#   nqmS    river R(S1) + a fixed log-grid of modes, non-negative weights (NNLS), exponents p, q chosen on the fit set
#   fsl     functional scaling law (Li et al. NeurIPS 2025): river R(S1) + C sum_k eta_k^2 K(S1(t) - S1(k+1)), power-law forgetting
#           kernel K(r) = (r + r0)^-gamma in intrinsic time (gamma = 2 - 1/beta for capacity beta)
# Usage: python -I k_models.py <results_dir> <ref_tag> <fit_tags,comma> <test_tags,comma> <out.json> [base_tag] [eta_ref]
#   a test tag may be a pair "run:reference" (e.g. an annealed Kovacs branch against the annealed reference branch, or a
#   seed's Kovacs run against that seed's reference); the excursion and its prediction are then taken against that run
import os, re, sys, json, numpy as np, pandas as pd
from scipy.optimize import least_squares, nnls
from scipy.signal import lfilter

R, ref_tag = sys.argv[1], sys.argv[2]
fit_tags = [t for t in sys.argv[3].split(",") if t]; test_tags = [t for t in sys.argv[4].split(",") if t]
out = sys.argv[5]; base_tag = sys.argv[6] if len(sys.argv) > 6 else "base"
ER = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-3                # reference learning rate (unit of eta)


def meta(tag): return json.load(open(f"{R}/{tag}_meta.json"))
def evals(tag):
    d = pd.read_csv(f"{R}/{tag}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]); return d.step.values, d.eval_loss.values


bm = meta(base_tag); WU = bm["phases"][0]["end"]


def parent(m):
    """run whose checkpoint this run started from (None for base checkpoints), so branched runs inherit its history"""
    ini = os.path.basename(m["args"].get("init") or "")[:-3]
    par = re.sub(r"_(end\d+|s\d+)$", "", ini)
    return par if par and par != ini and os.path.exists(f"{R}/{par}_meta.json") else None


def history(tag, until):
    """per-step lr from step 0 to `until` (exclusive), in units of ER: base warmup + constant (or the history of the run
    the checkpoint came from), then the run's own phases"""
    m = meta(tag); par = parent(m)
    if par: h = history(par, until); h[m.get("step0", 0):] = 1.0
    else: h = np.ones(until); h[:WU] = (np.arange(WU) + 1) / WU
    spec = json.loads(m["args"]["sched"])
    for ph in m["phases"]:
        a0, a1 = ph["start"], min(ph["end"], until)
        if a1 > a0 and (m.get("step0", 0) > 0 or ph["phase"] > 0):
            sp = spec[ph["phase"]]
            if sp.get("decay"): h[a0:a1] = ph["lr"] / ER * np.maximum(0.0, 1.0 - np.arange(a1 - a0) / sp["steps"])
            else: h[a0:a1] = ph["lr"] / ER
    return h


def ref_history(until): return history(ref_tag, until)


# ------------------------------------------------------------------ models (all vectorised over the eval steps)
def S1_of(h): return np.r_[0, np.cumsum(h)]                          # S1[t] = area before step t


def tissue(h, steps, th):
    L0, lA, al, C, lam = th
    S1 = S1_of(h)[steps]; d = np.r_[0, -np.diff(h)]; d[:WU] = 0       # decrements (warmup excluded)
    m = lfilter([1.0], [1.0, -lam], d)                                # m_i = lam m_{i-1} + d_i
    S2 = np.r_[0, np.cumsum(m)][steps]
    return L0 + np.exp(lA) * S1 ** (-al) - C * S2


def mpl(h, steps, th):
    L0, lA, al, lB, lC, be, ga = th
    SW = h[:WU].sum(); S = np.r_[0, np.cumsum(h)]; S1 = S[steps] - S[WU] + SW
    out_ = L0 + np.exp(lA) * np.maximum(S1, 1e-9) ** (-al)
    d = np.r_[0, -np.diff(h)]; ks = np.nonzero(np.abs(d) > 1e-12)[0]; ks = ks[ks >= WU]
    LD = np.zeros(len(steps))
    for k in ks:
        Sk = S[steps] - S[k]; ok = steps > k
        x = h[k] ** (-ga) * Sk[ok]
        LD[ok] += d[k] * (1 - (np.exp(lC) * x + 1) ** (-be))
    return out_ - np.exp(lB) * LD


def modes_u(h, rates, steps, p_=1.0, q_=1.0):
    """excess of each mode after `steps` updates, in units of the eta=ER equilibrium: on a constant-lr segment u relaxes at rate
    eta^q * r towards eta^p (eta in units of ER), exactly; the warmup ramp is treated as 20 constant pieces"""
    hh = h.copy(); c = max(WU // 20, 1)
    for i in range(0, WU, c): hh[i:i + c] = h[i:i + c].mean()
    idx = np.r_[0, np.nonzero(np.diff(hh))[0] + 1, len(hh)]
    steps = np.asarray(steps); U = np.zeros((len(steps), len(rates))); u = np.zeros(len(rates))
    for a, b in zip(idx[:-1], idx[1:]):
        e = hh[a]; eq, k = e ** p_, (e ** q_) * rates
        sel = (steps > a) & (steps <= b)
        if sel.any(): U[sel] = eq + (u - eq) * np.exp(-np.outer(steps[sel] - a, k))
        u = eq + (u - eq) * np.exp(-(b - a) * k)
    return U


def river(h, steps, th):
    L0, lA, al, lS0 = th
    return L0 + np.exp(lA) * (S1_of(h)[steps] + np.exp(lS0)) ** (-al)


def nqm2(h, steps, th):
    rv = river(h, steps, th[:4]); c1, c2, lr1, lr2 = th[4:]
    U = modes_u(h, np.exp([lr1, lr2]), steps)
    return rv + c1 * U[:, 0] + c2 * U[:, 1]


def nqm2pq(h, steps, th):
    """two modes with free exponents: equilibrium excess ~ eta^p, clock ~ eta^q (the theorem's g and k)"""
    rv = river(h, steps, th[:4]); c1, c2, lr1, lr2, p_, q_ = th[4:]
    U = modes_u(h, np.exp([lr1, lr2]), steps, p_, q_)
    return rv + c1 * U[:, 0] + c2 * U[:, 1]


def fsl_noise(h, steps, ga, r0):
    """sum_{k<t} eta_k^2 (S1(t) - S1(k+1) + r0)^-ga at every t in steps (eta in units of ER): warmup steps summed directly,
    constant-lr segments in closed form with the Euler-Maclaurin end correction"""
    S = S1_of(h); steps = np.asarray(steps); St = S[steps]; kw = np.arange(WU)
    N = ((h[kw] ** 2)[None, :] * (np.maximum(St[:, None] - S[kw + 1][None, :], 0) + r0) ** (-ga)
         * (kw[None, :] < steps[:, None])).sum(1)
    idx = np.r_[WU, WU + np.nonzero(np.diff(h[WU:]))[0] + 1, len(h)]
    j = np.arange(32)                                                  # the 32 most recent steps of a segment: exact sum
    for a, b in zip(idx[:-1], idx[1:]):
        e = h[a]; bb = np.minimum(steps, b); n = bb - a; ok = n > 0
        if e <= 0 or not ok.any(): continue
        D0 = St[ok] - S[bb[ok]] + r0; n = n[ok]
        N[ok] += e * e * ((D0[:, None] + e * j[None, :]) ** (-ga) * (j[None, :] < n[:, None])).sum(1)
        rest = n > len(j); D0, D1 = D0[rest] + e * len(j), D0[rest] + e * n[rest]
        N[np.nonzero(ok)[0][rest]] += e * e * ((D0 ** (1 - ga) - D1 ** (1 - ga)) / (e * (ga - 1)) + 0.5 * (D0 ** (-ga) - D1 ** (-ga)))
    return N


def fsl(h, steps, th):
    return river(h, steps, th[:4]) + np.exp(th[4]) * fsl_noise(h, steps, th[5], np.exp(th[6]))


# ------------------------------------------------------------------ data
def collect(tags):
    D = []
    for tag in tags:
        st, y = evals(tag)
        keep = st >= (1000 if tag == base_tag else 0)
        st, y = st[keep], y[keep]
        D.append((tag, st.astype(int), y, history(tag, int(st.max()) + 1)))
    return D

FIT = collect([base_tag, ref_tag] + fit_tags)
pairs = [(t.split(":") + [ref_tag])[:2] for t in test_tags]
TEST = [d + (rt,) for d, (_, rt) in zip(collect([t for t, _ in pairs]), pairs)]
REFS = {}
for rt in {rt for _, rt in pairs}:
    s_, y_ = evals(rt); REFS[rt] = (s_, y_, history(rt, int(s_.max()) + 1))


def resid(model, th, D):
    return np.concatenate([model(h, st, th) - y for _, st, y, h in D])


res = {"fit_tags": fit_tags, "test_tags": test_tags}
fits = {}
x0 = {"tissue": ([3.0, np.log(2.0), 0.3, 0.05, 0.99], ([0, -10, 0.01, 0, 0.5], [10, 10, 3, 10, 0.99999])),
      "mpl": ([3.0, np.log(2.0), 0.3, np.log(0.3), np.log(0.05), 0.5, 0.5], ([0, -10, 0.01, -10, -15, 0.01, 0.0], [10, 10, 3, 5, 5, 5, 3])),
      "nqm2": ([3.0, np.log(2.0), 0.3, np.log(100.0), 0.05, 0.1, np.log(0.1), np.log(0.003)],
               ([0, -10, 0.01, -5, -2, -2, np.log(1e-5), np.log(1e-5)], [10, 10, 3, 15, 2, 2, np.log(3.0), np.log(3.0)])),
      "fsl": ([3.0, np.log(2.0), 0.3, np.log(100.0), np.log(0.01), 1.5, np.log(10.0)],
              ([0, -10, 0.01, -5, -15, 1.02, np.log(0.1)], [10, 10, 3, 15, 5, 4.0, np.log(1e4)]))}
x0["nqm2pq"] = (x0["nqm2"][0] + [1.0, 1.0], (x0["nqm2"][1][0] + [0.1, 0.1], x0["nqm2"][1][1] + [3.0, 3.0]))
for name, model in (("tissue", tissue), ("mpl", mpl), ("nqm2", nqm2), ("fsl", fsl), ("nqm2pq", nqm2pq)):
    th0, bounds = x0[name]
    r = least_squares(lambda th: resid(model, th, FIT), th0, bounds=bounds, loss="soft_l1", f_scale=0.01, max_nfev=4000)
    fits[name] = (model, r.x); res[f"{name}_params"] = r.x.tolist(); res[f"{name}_fit_rmse"] = float(np.sqrt(np.mean(resid(model, r.x, FIT) ** 2)))
    try:                                                              # parameter s.e. from the Gauss-Newton covariance
        J = r.jac; dof = max(len(r.fun) - len(r.x), 1); cov = np.linalg.pinv(J.T @ J) * (2 * r.cost / dof)
        res[f"{name}_se"] = np.sqrt(np.clip(np.diag(cov), 0, None)).tolist()
    except Exception:
        pass
    res[f"{name}_fit_by_tag"] = {tag: float(np.sqrt(np.mean((model(h, st, r.x) - y) ** 2))) for tag, st, y, h in FIT}
    print(name, "fit rmse", round(res[f"{name}_fit_rmse"], 5), np.round(r.x, 4).tolist(), flush=True)
    print("   by tag", {k: round(v, 4) for k, v in res[f"{name}_fit_by_tag"].items()}, flush=True)

# spectral NQM: river fitted jointly, modes on a grid with NNLS; p, q by grid search on the fit set
rth = fits["nqm2"][1][:4]
grid = np.logspace(-4.5, 0.3, 26)
best = None
for p_ in (0.5, 1.0, 1.5):
    for q_ in (0.5, 1.0, 1.5):
        X, Y = [], []
        for _, st, y, h in FIT:
            X.append(np.c_[modes_u(h, grid, st, p_, q_), np.ones(len(st))]); Y.append(y - river(h, st, rth))
        X, Y = np.vstack(X), np.concatenate(Y)
        X[:, -1] = 1.0; Xp = np.c_[X, -X[:, -1]]                     # free-sign constant via two non-negative columns
        w, _ = nnls(Xp, Y); rm = float(np.sqrt(np.mean((Xp @ w - Y) ** 2)))
        if best is None or rm < best[0]: best = (rm, p_, q_, w)
rmS, pS, qS, wS = best
def nqmS(h, st, th=None):
    return river(h, st, rth) + np.c_[modes_u(h, grid, st, pS, qS), np.ones(len(st)), -np.ones(len(st))] @ wS
fits["nqmS"] = (nqmS, None); res["nqmS_fit_rmse"] = rmS; res["nqmS_pq"] = [pS, qS]
res["nqmS_fit_by_tag"] = {tag: float(np.sqrt(np.mean((nqmS(h, st) - y) ** 2))) for tag, st, y, h in FIT}
res["fit_curves"] = {tag: {"step": st[::4].tolist(), "obs": y[::4].tolist(),
                           **{nm: f[0](h, st[::4], f[1]).tolist() for nm, f in fits.items()}} for tag, st, y, h in FIT}
res["nqmS_spectrum"] = [[float(1 / g), float(w)] for g, w in zip(grid, wS[:-2]) if w > 0]   # (tau in ref steps, weight)
print("nqmS fit rmse", round(rmS, 5), "p,q", pS, qS, "modes", [[round(a), round(b, 4)] for a, b in res["nqmS_spectrum"]], flush=True)

# ------------------------------------------------------------------ blind predictions
for tag, st, y, h, rt in TEST:
    m = meta(tag); ph = [p for p in m["phases"] if p.get("matched_at")]
    rel = ph[0]["matched_at"] if ph else None
    rs, ry, rh = REFS[rt]
    st2 = st[(st >= rs.min()) & (st <= rs.max())]
    obs = np.interp(st2, st, y) - np.interp(st2, rs, ry)
    rec = {"matched_at": rel, "ref": rt, "final_obs": float(obs[-1])}
    for name, (model, th) in fits.items():
        pa = model(h[: st2.max() + 1], st2, th); pr = pa - model(rh[: st2.max() + 1], st2, th)
        e = {"rmse_abs": float(np.sqrt(np.mean((pa - np.interp(st2, st, y)) ** 2))),   # registered: absolute loss, whole run
             "rmse_diff": float(np.sqrt(np.mean((pr - obs) ** 2))),                  # secondary: loss difference to the reference
             "final_pred": float(pr[-1])}
        if rel:
            a = st2 >= rel; t = st2[a] - rel; o = obs[a]; pp = pr[a]
            sg = 1.0 if o[np.argmax(np.abs(o[: max(3, len(o) // 4)]))] > 0 else -1.0
            os_ = pd.Series(o).rolling(3, center=True, min_periods=1).mean().values
            e.update({"nrmse_after": float(np.sqrt(np.mean((pp - o) ** 2)) / max(np.ptp(os_), 1e-9)),
                      "peak_obs": float(os_[np.argmax(sg * os_)]), "t_peak_obs": int(t[np.argmax(sg * os_)]),
                      "peak_pred": float(pp[np.argmax(sg * pp)]), "t_peak_pred": int(t[np.argmax(sg * pp)]),
                      "curve": [[int(a_), float(b_), float(c_)] for a_, b_, c_ in zip(t, o, pp)]})
        rec[name] = e
        print(tag, name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k != "curve"}), flush=True)
    res[tag if rt == ref_tag else f"{tag}:{rt}"] = rec
json.dump(res, open(out, "w"), indent=1)
