# E10 excursions of run pairs from their evaluation logs: dL(t) = L_run(t) - L_base(t) on the matched set A (eval_loss) and,
# where logged, the untouched set B (eval_loss_b), against steps since the first common evaluation (or since a given step),
# with the statistics fixed in PREREG.md (peak = max of the 3-evaluation running mean over 10-150 steps; ext = the signed
# extremum there, secondary; area = integral over 0-300 steps; late = mean over 1000-2000 steps, or over 1000-1500 steps for
# the 1500-step continuations; a window the pair does not cover gives a missing value) and the relative weight-norm difference
# |w_run|/|w_base| - 1 at the start and at the end.
# Usage: python -I k_pairs.py <results_dir> <run:base[@t0],run:base,...> <out.json> [late_end]
#   late_end (1500 or 2000) fixes the late window as 1000-late_end for every pair; the pair must then cover it (within one
#   sparse evaluation interval of 60 steps), else the late offset is missing. Without it the window follows the coverage.
import sys, json, numpy as np, pandas as pd

R, pairs, out = sys.argv[1], [p for p in sys.argv[2].split(",") if p], sys.argv[3]
LATE_END = int(sys.argv[4]) if len(sys.argv) > 4 else None


def ld(tag):
    return pd.read_csv(f"{R}/{tag}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")


def stats(t, y):   # each statistic only over its declared window, never a substitute: peak 10-150, area 0-300, late
    ys = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values      # 1000-2000 (1000-1500 for 1500-step runs)
    if t.max() < 150: return {}
    w = (t >= 10) & (t <= 150); a = (t >= 0) & (t <= 300)
    if LATE_END is None:
        late = (t >= 1000) & (t <= (2000 if t.max() >= 2000 else 1500)) if t.max() >= 1500 else None
    else:                                                                       # declared window, never a substitute
        late = (t >= 1000) & (t <= LATE_END) if t.max() >= LATE_END - 60 else None
    sg = 1.0 if ys[w][np.argmax(np.abs(ys[w]))] > 0 else -1.0
    i = np.nonzero(w)[0][np.argmax(ys[w])]; j = np.nonzero(w)[0][np.argmax(sg * ys[w])]
    return {"peak": float(ys[i]), "t_peak": float(t[i]), "ext": float(ys[j]), "t_ext": float(t[j]),
            "area": float(np.trapezoid(y[a], t[a])) if t.max() >= 300 else None,
            "late": None if late is None else float(y[late].mean()),
            "late_window": None if late is None else [float(t[late].min()), float(t[late].max())]}


res = {}
for p in pairs:
    run, base = p.split(":"); t0 = None
    if "@" in base: base, t0 = base.split("@"); t0 = int(t0)
    try:
        d, b = ld(run), ld(base)
    except FileNotFoundError as e:
        print(p, "missing", e.filename); continue
    s0 = t0 if t0 is not None else int(max(d.step.min(), b.step.min()))
    d = d[(d.step >= s0) & (d.step <= b.step.max())]
    t = (d.step.values - s0).astype(float); rec = {"t0": s0, "t": t.tolist()}
    for col, key in (("eval_loss", "A"), ("eval_loss_b", "B")):
        if col in d and col in b and d[col].notna().any() and b[col].notna().any():
            y = d[col].values - np.interp(d.step.values, b.step.values, b[col].values)
            if np.isfinite(y).all(): rec[key] = y.tolist(); rec[f"stats_{key}"] = stats(t, y)
    if "wnorm" in d and "wnorm" in b:
        wr = d.wnorm.values / np.interp(d.step.values, b.step.values, b.wnorm.values) - 1
        rec["wnorm_rel"] = wr.tolist(); rec["wnorm_rel_start"] = float(wr[0]); rec["wnorm_rel_end"] = float(wr[-1])
    res[p] = rec
    print(p, "A", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.get("stats_A", {}).items()}),
          "B", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.get("stats_B", {}).items()}),
          "w", round(rec.get("wnorm_rel_start", np.nan), 4), round(rec.get("wnorm_rel_end", np.nan), 4), flush=True)
json.dump(res, open(out, "w"))
