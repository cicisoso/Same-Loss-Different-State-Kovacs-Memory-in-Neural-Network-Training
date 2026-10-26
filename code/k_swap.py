# E10 optimizer-state transplant as a 2x2 factorial: at the matched step the Kovacs weights (K) and the reference weights (R)
# are each continued with their own AdamW state and with the other run's state, on identical data. With Y(w, s) the held-out
# loss of weights w with state s, all curves are taken relative to Y(R, R):
#   own      Y(K, K) - Y(R, R)    the Kovacs excursion as observed
#   weights  Y(K, R) - Y(R, R)    Kovacs weights with the reference state
#   state    Y(R, K) - Y(R, R)    reference weights with the Kovacs state
# and decomposed into main effects and their interaction,
#   W = [Y(K,K) + Y(K,R) - Y(R,K) - Y(R,R)] / 2,  S = [Y(K,K) + Y(R,K) - Y(K,R) - Y(R,R)] / 2,
#   I = Y(K,K) - Y(K,R) - Y(R,K) + Y(R,R).
# Statistics as registered (PREREG.md): peak = max of the 3-evaluation running mean over 10-150 steps (signed extremum as a
# secondary column), area over 0-300 steps, late = mean over 1000-1500 steps (the transplant continuations are 1500 steps
# long, so this late window differs from the 1000-2000 window of the 2500-step releases). Curves are compared only on the
# steps covered by all four branches; a statistic whose window is not covered is missing.
# Usage: python -I k_swap.py <results_dir> [out.json]
import sys, json, numpy as np, pandas as pd

R = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else f"{R}/k_swap.json"
ld = lambda t: pd.read_csv(f"{R}/{t}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
TAGS = {"KK": "sw_kp_own", "KR": "sw_kp_refopt", "RK": "sw_ref_kpopt", "RR": "sw_ref_own"}


def stats(t, y):
    ys = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values
    w = (t >= 10) & (t <= 150); a = (t >= 0) & (t <= 300); l = (t >= 1000) & (t <= 1500)
    out = {"peak": None, "t_peak": None, "ext": None, "t_ext": None, "area": None, "late": None}
    if t.max() >= 150:
        i = np.nonzero(w)[0][np.argmax(ys[w])]; j = np.nonzero(w)[0][np.argmax(np.abs(ys[w]))]
        out.update(peak=float(ys[i]), t_peak=float(t[i]), ext=float(ys[j]), t_ext=float(t[j]))
    if t.max() >= 300: out["area"] = float(np.trapezoid(y[a], t[a]))
    if t.max() >= 1500: out["late"] = float(y[l].mean())
    return out


D = {k: ld(v) for k, v in TAGS.items()}
base = D["RR"]; t0 = int(base.step.min()); res = {"t0": t0, "tags": TAGS}
for col in ("eval_loss", "eval_loss_b"):
    if any(col not in d or d[col].isna().all() for d in D.values()): continue
    lo, hi = max(d.step.min() for d in D.values()), min(d.step.max() for d in D.values())   # common support
    st = base.step.values; st = st[(st >= lo) & (st <= hi)]; t = (st - t0).astype(float)
    Y = {k: np.interp(st, d.step.values, d[col].values) for k, d in D.items()}
    curves = {"own": Y["KK"] - Y["RR"], "weights": Y["KR"] - Y["RR"], "state": Y["RK"] - Y["RR"],
              "W": (Y["KK"] + Y["KR"] - Y["RK"] - Y["RR"]) / 2, "S": (Y["KK"] + Y["RK"] - Y["KR"] - Y["RR"]) / 2,
              "I": Y["KK"] - Y["KR"] - Y["RK"] + Y["RR"]}
    res[col] = {"t": t.tolist(), "curves": {k: v.tolist() for k, v in curves.items()},
                "stats": {k: stats(t, v) for k, v in curves.items()}}
    for k, v in res[col]["stats"].items():
        print(col, k, json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}), flush=True)
json.dump(res, open(out, "w"))
