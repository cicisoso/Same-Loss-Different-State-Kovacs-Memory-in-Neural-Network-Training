# E10 Boltzmann superposition: the post-release excursion of a Kovacs run predicted from single learning-rate steps alone,
# with no parameter fitted to any Kovacs run. If the hidden modes relax linearly in the intrinsic clock s (ds = eta/eta_ref dt)
# towards levels g = eta/eta_ref, a single step from g = 1 to g produces dE(s) = (g - 1) R(s), and any piecewise-constant
# history with level changes dg_k at intrinsic times s_k before release gives dE(sig) = sum_k dg_k R(s_k + sig). For heat
# (g_h, s_a), cool (g_l, s_c) and release to 1:
#   dE(sig) = (g_h - 1) R(s_a + s_c + sig) + (g_l - g_h) R(s_c + sig) + (1 - g_l) R(sig).
# Conditioning on the match (no excursion at release), the prediction is dL(sig) = [dE(sig) - dE(0)] + [dV(sig) - dV(0)],
# where dV is the river: the reference's own loss read at the run's cumulative learning rate (a power law fitted to the
# reference, L0 + A (S + S0)^-alpha). R(s) is the river-corrected per-unit response of the cooling steps, averaged in
# intrinsic time; beyond the longest measured s it is continued logarithmically with the slope of its last decade.
# Usage: python -I k_super.py <results_dir> <step_ref> <cooling_steps,comma> <tests kov:ref,comma> <out.json> [eta_ref]
#        [heating_steps,comma] [norm_exp]   (heating steps are only reported, never used for prediction)
# Exploratory option norm_exp = a > 0: the Kovacs run's level and clock use the effective learning rate
# eta_eff(t) = eta(t) (|w_ref(t)| / |w(t)|)^a, with |w| the norm of the decayed weights logged at every evaluation (the
# angular step of a scale-invariant weight under Adam is ~ eta/|w|, a = 1; under SGD ~ eta/|w|^2, a = 2); the excursion is
# then the Stieltjes sum dE(t) = sum_k [g(k) - g(k-1)] R(s(t) - s(k)) over every training step k.
import sys, json, numpy as np, pandas as pd
from scipy.optimize import curve_fit

R, step_ref, cool = sys.argv[1], sys.argv[2], [t for t in sys.argv[3].split(",") if t]
tests = [(t.split(":") + [step_ref])[:2] for t in sys.argv[4].split(",") if t]
out = sys.argv[5]; ER = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-3
heat = [t for t in sys.argv[7].split(",") if t] if len(sys.argv) > 7 else []
NA = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0


def ld(tag):
    d = pd.read_csv(f"{R}/{tag}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
    return d.step.values.astype(float), d.eval_loss.values


def meta(tag): return json.load(open(f"{R}/{tag}_meta.json"))


def river_fit(tag):
    """power law L0 + A (S + S0)^-alpha fitted to a constant-eta_ref reference against steps since its first evaluation"""
    st, y = ld(tag); S = st - st[0]
    f = lambda S, L0, lA, al, lS0: L0 + np.exp(lA) * (S + np.exp(lS0)) ** (-al)
    p, _ = curve_fit(f, S, y, p0=[y.min() - 0.3, np.log(1.0), 0.5, np.log(1000.0)], maxfev=20000,
                     bounds=([0, -10, 0.01, np.log(1.0)], [20, 10, 5, np.log(1e6)]))
    return (lambda S: f(np.asarray(S, float), *p)), st[0], p.tolist(), float(np.sqrt(np.mean((f(S, *p) - y) ** 2)))


# ------------------------------------------------------------------ per-unit step response R(s)
Vs, s0_step, vpar, vrmse = river_fit(step_ref)
rs_, ry_ = ld(step_ref)
grid = np.logspace(-1, 4, 101)                                        # intrinsic time grid (reference steps)
res = {"step_ref": step_ref, "river_params": vpar, "river_rmse": vrmse, "steps": {}}


def per_unit(tag):
    st, y = ld(tag); g = meta(tag)["phases"][0]["lr"] / ER
    t = st - s0_step; k = (st >= rs_.min()) & (st <= rs_.max())
    dl = y[k] - np.interp(st[k], rs_, ry_); t = t[k]
    dv = Vs(g * t) - Vs(t)                                            # river: run behind (cooling) or ahead (heating)
    return g, t, (dl - dv) / (g - 1), dl / (g - 1)


def on_grid(g, t, pu):
    """per-unit response on the intrinsic grid (3-point running mean in steps first); NaN outside the measured range"""
    ps = pd.Series(pu).rolling(3, center=True, min_periods=1).mean().values
    s = g * t; m = s > 0
    return np.where((grid >= s[m].min()) & (grid <= s[m].max()), np.interp(grid, s[m], ps[m]), np.nan)


Rc = []
for tag in cool + heat:
    g, t, pu, raw = per_unit(tag); G = on_grid(g, t, pu)
    res["steps"][tag] = {"eta": g, "t": t.tolist(), "per_unit": pu.tolist(), "per_unit_raw": raw.tolist(), "grid": G.tolist(),
                         "use": tag in cool}
    if tag in cool: Rc.append(G)
Rc = np.array(Rc); n_av = np.sum(np.isfinite(Rc), 0)
Rm = np.where(n_av > 0, np.nansum(Rc, 0) / np.maximum(n_av, 1), np.nan)
ok = np.isfinite(Rm); smax = grid[ok].max(); smin = grid[ok].min()
last = ok & (grid >= smax / 10)
w_tail = float(np.polyfit(np.log(grid[last]), Rm[last], 1)[0])       # slope per e-fold over the last measured decade
mid = ok & (grid >= 1) & (grid <= 50)
w_mid = float(np.polyfit(np.log(grid[mid]), Rm[mid], 1)[0]) if mid.sum() > 3 else None
res["R"] = {"s": grid.tolist(), "R": Rm.tolist(), "n": n_av.tolist(), "s_max": float(smax), "w_tail": w_tail, "w_mid": w_mid}
print("R(s): measured to s =", round(smax), "slope per e-fold, last decade", round(w_tail, 4), "s in [1,50]",
      None if w_mid is None else round(w_mid, 4), flush=True)


def Rfun(s):
    s = np.asarray(s, float); v = np.interp(np.log(np.maximum(s, smin)), np.log(grid[ok]), Rm[ok])
    v = np.where(s > smax, Rm[ok][-1] + w_tail * np.log(np.maximum(s, smax) / smax), v)
    return np.where(s <= 0, 0.0, np.where(s < smin, Rm[ok][0] * s / smin, v))


# ------------------------------------------------------------------ predictions for Kovacs runs
def protocol(tag):
    """levels and intrinsic durations of the phases before release, the release step and the cumulative-lr offset"""
    m = meta(tag); ph = m["phases"]; rel = [p for p in ph if p.get("matched_at")]
    if not rel: return None
    T = rel[0]["matched_at"]; pre = [p for p in ph if p["start"] < T]
    lev = [(p["lr"] / ER, (min(p["end"], T) - p["start"]) * p["lr"] / ER, min(p["end"], T) - p["start"]) for p in pre]
    dA = sum(s for _, s, _ in lev) - sum(n for _, _, n in lev)        # extra cumulative lr at release (reference steps)
    return lev, T, dA, ph[0]["start"]


def score(t, o, p):
    os_ = pd.Series(o).rolling(3, center=True, min_periods=1).mean().values; ps_ = p
    w = (t >= 10) & (t <= 150)
    if not w.any(): return {}
    sg = 1.0 if os_[w][np.argmax(np.abs(os_[w]))] > 0 else -1.0
    i = np.nonzero(w)[0][np.argmax(sg * os_[w])]; j = np.nonzero(w)[0][np.argmax(sg * ps_[w])]
    sp = 1.0 if ps_[w][np.argmax(np.abs(ps_[w]))] > 0 else -1.0
    a = t <= 2500
    nr = float(np.sqrt(np.mean((p[a] - o[a]) ** 2)) / max(np.ptp(os_[a]), 1e-9))
    hr = float(ps_[j] / os_[i]); tr = float(t[j] / max(t[i], 1))
    return {"sign_obs": sg, "sign_pred": sp, "peak_obs": float(os_[i]), "t_peak_obs": float(t[i]), "peak_pred": float(ps_[j]),
            "t_peak_pred": float(t[j]), "height_ratio": hr, "time_ratio": tr, "nrmse": nr,
            "success": bool(sg == sp and 1 / 1.5 < hr < 1.5 and 1 / 1.5 < tr < 1.5 and nr < 0.3)}


res["tests"] = {}
for kov, ref in tests:
    pr = protocol(kov)
    if pr is None: print(kov, "no match"); continue
    lev, T, dA, start = pr
    ks, ky = ld(kov); qs, qy = ld(ref)
    a = (ks >= T) & (ks <= qs.max()); sig = ks[a] - T
    obs = ky[a] - np.interp(ks[a], qs, qy)
    Vq = river_fit(ref)[0] if ref != step_ref else Vs; tq = T - qs[0]
    # level changes before release: from 1 to lev[0], lev[0] to lev[1], ..., lev[-1] to 1 (release)
    gs = [1.0] + [g for g, _, _ in lev] + [1.0]; back = np.cumsum([s for _, s, _ in lev][::-1])[::-1]   # intrinsic time before release
    times = list(back) + [0.0]
    dE = lambda x: sum((gs[k + 1] - gs[k]) * Rfun(times[k] + x) for k in range(len(times)))
    if NA > 0:                                                        # per-step effective levels from the logged norms
        km = pd.read_csv(f"{R}/{kov}_eval.tsv", sep="\t").sort_values("step"); qm = pd.read_csv(f"{R}/{ref}_eval.tsv", sep="\t").sort_values("step")
        n_end = int(min(ks.max(), qs.max())); stp = np.arange(start, n_end); lr = np.ones(len(stp))
        for p_ in meta(kov)["phases"]:
            lr[(stp >= p_["start"]) & (stp < p_["end"])] = p_["lr"] / ER
        ratio = np.interp(stp, qm.step, qm.wnorm) / np.interp(stp, km.step, km.wnorm)
        g = lr * ratio ** NA; scum = np.r_[0, np.cumsum(g)]; dg = np.diff(np.r_[1.0, g])
        nz = np.nonzero(np.abs(dg) > 1e-9)[0]
        def dE_t(tt):                                                 # excursion at absolute steps tt (after release)
            i = (np.asarray(tt) - start).astype(int)
            return np.array([np.sum(dg[nz[nz < j]] * Rfun(scum[j] - scum[nz[nz < j]])) for j in i])
        base = dE_t(np.array([T]))[0]
        dE = lambda x: dE_t(T + np.minimum(np.asarray(x), n_end - T - 1)) if np.ndim(x) else dE_t(np.array([T + x]))[0]
        res.setdefault("norm_ratio_at_release", {})[kov] = float(ratio[T - start])
    dV = lambda x: Vq(tq + x + dA) - Vq(tq + x)
    pred = (dE(sig) - dE(np.zeros(1))[0]) + (dV(sig) - dV(np.zeros(1))[0])
    sc = score(sig, obs, pred)
    res["tests"][f"{kov}:{ref}"] = {"levels": lev, "release": int(T), "dA": float(dA), "t": sig.tolist(), "obs": obs.tolist(),
                                    "pred": pred.tolist(), "pred_modes": (dE(sig) - dE(np.zeros(1))[0]).tolist(), **sc}
    print(kov, ref, "dA", round(dA), json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in sc.items()}), flush=True)
json.dump(res, open(out, "w"))
