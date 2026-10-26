# E10 sensitivity of the blind contest to the optimizer of the parametric fits (Codex rounds 4-5; not the frozen contest).
# k_models.py fits each parametric model (single-timescale law, multi-power law, two-mode model with fixed and with free
# exponents, functional scaling law) once, from a fixed starting point, with a soft-l1 loss on the fit set. Here each is
# refitted on the SAME fit set from that starting point and from N-1 random points drawn uniformly within the same bounds
# (same loss, scale and evaluation budget); the fit-set cost, convergence status and parameters of every start are kept,
# and the best start is asked the same questions as the frozen fit: predicted excursion of each test run (peak and its lag
# after release, final predicted gap). The spectral model (non-negative least squares on a fixed grid, exponents by grid
# search) is a convex fit and is not repeated. Model definitions and data handling are those of k_models.py, copied.
# Usage: python -I k_models_ms.py <results_dir> <ref_tag> <fit_tags,comma> <test_tags,comma> <out.json> [n_starts=16] [workers=16]
import os, re, sys, json, numpy as np, pandas as pd
from scipy.optimize import least_squares, nnls
from scipy.signal import lfilter

R, ref_tag = sys.argv[1], sys.argv[2]
fit_tags = [t for t in sys.argv[3].split(",") if t]; test_tags = [t for t in sys.argv[4].split(",") if t]
out = sys.argv[5]; NS = int(sys.argv[6]) if len(sys.argv) > 6 else 16; NW = int(sys.argv[7]) if len(sys.argv) > 7 else 16
base_tag, ER = "base", 1e-3


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


x0 = {"tissue": ([3.0, np.log(2.0), 0.3, 0.05, 0.99], ([0, -10, 0.01, 0, 0.5], [10, 10, 3, 10, 0.99999])),
      "mpl": ([3.0, np.log(2.0), 0.3, np.log(0.3), np.log(0.05), 0.5, 0.5], ([0, -10, 0.01, -10, -15, 0.01, 0.0], [10, 10, 3, 5, 5, 5, 3])),
      "nqm2": ([3.0, np.log(2.0), 0.3, np.log(100.0), 0.05, 0.1, np.log(0.1), np.log(0.003)],
               ([0, -10, 0.01, -5, -2, -2, np.log(1e-5), np.log(1e-5)], [10, 10, 3, 15, 2, 2, np.log(3.0), np.log(3.0)])),
      "fsl": ([3.0, np.log(2.0), 0.3, np.log(100.0), np.log(0.01), 1.5, np.log(10.0)],
              ([0, -10, 0.01, -5, -15, 1.02, np.log(0.1)], [10, 10, 3, 15, 5, 4.0, np.log(1e4)]))}
x0["nqm2pq"] = (x0["nqm2"][0] + [1.0, 1.0], (x0["nqm2"][1][0] + [0.1, 0.1], x0["nqm2"][1][1] + [3.0, 3.0]))
MODELS = {"tissue": tissue, "mpl": mpl, "nqm2": nqm2, "fsl": fsl, "nqm2pq": nqm2pq}


def fit_one(job):
    name, i, th0 = job; lo, hi = x0[name][1]
    try:
        r = least_squares(lambda th: resid(MODELS[name], th, FIT), th0, bounds=(lo, hi), loss="soft_l1", f_scale=0.01, max_nfev=4000)
        return name, i, {"start": list(map(float, th0)), "x": r.x.tolist(), "cost": float(r.cost), "status": int(r.status),
                         "nfev": int(r.nfev), "rmse": float(np.sqrt(np.mean(resid(MODELS[name], r.x, FIT) ** 2)))}
    except Exception as ex:                                            # a start that fails to evaluate is recorded, not dropped
        return name, i, {"start": list(map(float, th0)), "error": repr(ex)[:200]}


def predict(model, th):
    out_ = {}
    for tag, st, y, h, rt in TEST:
        m = meta(tag); ph = [p for p in m["phases"] if p.get("matched_at")]; rel = ph[0]["matched_at"] if ph else None
        rs, ry, rh = REFS[rt]; st2 = st[(st >= rs.min()) & (st <= rs.max())]
        obs = np.interp(st2, st, y) - np.interp(st2, rs, ry)
        pr = model(h[: st2.max() + 1], st2, th) - model(rh[: st2.max() + 1], st2, th)
        e = {"final_obs": float(obs[-1]), "final_pred": float(pr[-1])}
        if rel:
            a = st2 >= rel; t = st2[a] - rel; pp = pr[a]; w = (t >= 10) & (t <= 150)
            e.update({"max_pred_10_150": float(pp[w].max()) if w.any() else None,
                      "t_max_pred": int(t[w][np.argmax(pp[w])]) if w.any() else None,
                      "pred_at_release": float(pp[0]), "hump_pred": bool(w.any() and pp[w].max() > max(pp[0], 0) + 1e-4)})
        out_[tag if rt == ref_tag else f"{tag}:{rt}"] = e
    return out_


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    rng = np.random.default_rng(0); jobs = []
    for name in MODELS:
        lo, hi = map(np.asarray, x0[name][1])
        jobs += [(name, 0, list(x0[name][0]))] + [(name, i, (lo + (hi - lo) * rng.uniform(0.02, 0.98, len(lo))).tolist())
                                                  for i in range(1, NS)]
    with ProcessPoolExecutor(NW) as ex:
        done = list(ex.map(fit_one, jobs))
    res = {"fit_tags": fit_tags, "test_tags": test_tags, "n_starts": NS}
    for name, model in MODELS.items():
        runs = [r for n, i, r in sorted(done, key=lambda z: z[1]) if n == name]
        ok = [r for r in runs if "cost" in r]; best = min(ok, key=lambda r: r["cost"])
        prim = runs[0]
        res[name] = {"starts": runs, "primary_cost": prim.get("cost"), "best_cost": best["cost"],
                     "primary_rmse": prim.get("rmse"), "best_rmse": best["rmse"], "best_is_primary": best is prim,
                     "n_within_1pct_of_best": int(sum(r["cost"] <= best["cost"] * 1.01 for r in ok)),
                     "n_failed": len(runs) - len(ok),
                     "pred_primary": predict(model, np.array(prim["x"])) if "x" in prim else None,
                     "pred_best": predict(model, np.array(best["x"]))}
        print(name, "cost primary", round(prim.get("cost", np.nan), 5), "best", round(best["cost"], 5), "rmse", round(prim.get("rmse", np.nan), 5),
              round(best["rmse"], 5), "near-best starts", res[name]["n_within_1pct_of_best"], "failed", res[name]["n_failed"], flush=True)
        for k_, v in res[name]["pred_best"].items():
            print("   best", k_, {q: (round(x, 4) if isinstance(x, float) else x) for q, x in v.items()}, flush=True)
    json.dump(res, open(out, "w"), indent=1)
