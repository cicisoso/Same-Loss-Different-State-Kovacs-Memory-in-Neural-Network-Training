# E10 peak-time law test (PREREG addendum): for each Kovacs run, the lag of the excursion maximum predicted from the protocol
# alone, sigma* = root of (g_h-g_l) A'(s_c+s) - (g_h-g_0) A'(s_c+s_a+s) - (g_0-g_l) A'(s) = 0 with A'(x) = -x^(-1-nu),
# g = (eta/eta_0)^p, intrinsic durations s = steps * (eta/eta_0)^q, compared with the observed peak: two-mode fit t* (the
# pre-registered statistic) and the maximum of the 3-evaluation running mean.
# The smoothed maximum is taken in the registered window 10-150 steps (t_max; the 1-600 window of the first version is kept
# as t_max_wide). Success is reported for both observed statistics, never for the better of the two. Two timing baselines
# calibrated on the exploratory pilot kp (fixed 2026-10-20 06:55 CST, before any withheld run): a constant lag of 41.3
# steps and a constant fraction 0.469 of the intrinsic cooling time (kp: t* = 41.3 after s_c = 88).
# Usage: python -I k_peaklaw.py <results_dir> <kov:ref,kov:ref,...> [nu=0] [p=1] [q=1] [out.json]
import sys, json, numpy as np, pandas as pd
from scipy.optimize import brentq, curve_fit

R = sys.argv[1]; pairs = [x.split(":") for x in sys.argv[2].split(",") if x]
nu = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; p = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
q = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0; out = sys.argv[6] if len(sys.argv) > 6 else f"{R}/k_peaklaw.json"
f2 = lambda t, A, ltf, lts, C: A * (np.exp(-t / np.exp(lts)) - np.exp(-t / np.exp(ltf))) + C


def sigma_star(gh, g0, gl, sa, sc):
    dA = lambda x: -x ** (-1 - nu)
    f = lambda s: (gh - gl) * dA(sc + s) - (gh - g0) * dA(sc + sa + s) - (g0 - gl) * dA(s)
    return brentq(f, 1e-6 * max(sc, 1e-3), 1e4 * (sc + sa + 1))


res = {}
for kt, rt in pairs:
    try:
        m = json.load(open(f"{R}/{kt}_meta.json"))
        d = pd.read_csv(f"{R}/{kt}_eval.tsv", sep="\t").dropna(subset=["eval_loss"])
        r = pd.read_csv(f"{R}/{rt}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
    except FileNotFoundError:
        continue
    ph = m["phases"]; mi = [i for i, x in enumerate(ph) if x.get("matched_at")]
    if not mi: continue
    i = mi[0]; TM = ph[i]["matched_at"]; e0 = ph[i + 1]["lr"] if i + 1 < len(ph) else 1e-3
    eh, ta, el, tc = ph[i - 1]["lr"], ph[i - 1]["end"] - ph[i - 1]["start"], ph[i]["lr"], TM - ph[i]["start"]
    gh, gl = (eh / e0) ** p, (el / e0) ** p; sa, sc = ta * (eh / e0) ** q, tc * (el / e0) ** q
    hot_first = eh > e0
    try:                                         # no sign change of dE/ds in the bracket: the relation predicts no maximum
        pred = sigma_star(gh, 1.0, gl, sa, sc) if hot_first else None; pred_note = None if hot_first else "cool-heat protocol"
    except ValueError:
        pred, pred_note = None, "undefined: no root of the peak equation"
    d = d[(d.step >= TM) & (d.step <= r.step.max())]
    t = (d.step - TM).values.astype(float); y = d.eval_loss.values - np.interp(d.step, r.step, r.eval_loss)
    ys = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values
    w = (t >= 1) & (t <= 600); tmax_wide = float(t[w][np.argmax(ys[w])])
    wr = (t >= 10) & (t <= 150); tmax = float(t[wr][np.argmax(ys[wr])])
    try:
        po, _ = curve_fit(f2, t, y, p0=[ys[w].max() * 1.5, np.log(10), np.log(300), 0.0], maxfev=50000)
        tf, ts = sorted([np.exp(po[1]), np.exp(po[2])]); tstar = tf * ts / (ts - tf) * np.log(ts / tf)
    except Exception:
        tstar = None
    if tstar is not None and not (np.isfinite(tstar) and tstar > 0): tstar = None      # finite, positive fitted lag only
    ok = lambda pr, ob: (bool(abs(np.log(pr / ob)) < np.log(1.5)) if (pr and ob and pr > 0 and ob > 0) else None)
    base = {"const": 41.3, "frac": 0.469 * sc}
    res[kt] = {"ref": rt, "eta_h": eh, "t_a": ta, "eta_l": el, "t_c": tc, "s_a": sa, "s_c": sc, "pred": pred, "pred_note": pred_note,
               "t_star": tstar, "t_max": tmax, "t_max_wide": tmax_wide, "peak": float(ys[wr].max()),
               "ok_t_star": ok(pred, tstar), "ok_t_max": ok(pred, tmax),
               **{f"base_{k}": v for k, v in base.items()},
               **{f"base_{k}_ok_t_star": ok(v, tstar) for k, v in base.items()}, **{f"base_{k}_ok_t_max": ok(v, tmax) for k, v in base.items()}}
    print(kt, json.dumps({k_: (round(v, 2) if isinstance(v, float) else v) for k_, v in res[kt].items()}), flush=True)
json.dump(res, open(out, "w"), indent=1)
