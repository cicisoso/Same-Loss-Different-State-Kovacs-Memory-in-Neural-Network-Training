# E10 endpoint reproducibility: area (0-300), late offset (1000-1500) and smoothed maximum (10-150) of the difference between
# two identical continuations of the same matched checkpoint (sham branch vs transplant own-state branch).
# Usage: python -I k_repro_ep.py <results_dir>
import sys, numpy as np, pandas as pd
R = sys.argv[1]
ld = lambda t: pd.read_csv(f"{R}/{t}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
for a_, b_ in (("nm_kp_sham", "sw_kp_own"), ("nm_ref_sham", "sw_ref_own")):
    A, B = ld(a_), ld(b_); t0 = A.step.min(); st = A.step.values; t = st - t0
    for col in ("eval_loss", "eval_loss_b"):
        d = A[col].values - np.interp(st, B.step.values, B[col].values)
        w = (t >= 0) & (t <= 300); l = (t >= 1000) & (t <= 1500); p = (t >= 10) & (t <= 150)
        ds = pd.Series(d).rolling(3, center=True, min_periods=1).mean().values
        print(a_, b_, col, "area %.3f late %.5f maxabs_smooth_10_150 %.5f" % (np.trapezoid(d[w], t[w]), d[l].mean(), np.abs(ds[p]).max()))
