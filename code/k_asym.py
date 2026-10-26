# E10 asymmetry of approach (Kovacs 1963): after an up-jump in learning rate the loss excess equilibrates faster than after a
# down-jump of the same log size. For each constant-lr run from the common checkpoint, dL(t) = loss - reference; the plateau is
# the mean dL over 200-300 steps after the jump; t50 / t80 are the first steps at which dL reaches 50% / 80% of the plateau
# (3-eval running mean). Noisy-quadratic picture: relaxation rate ~ eta, so t_x(up) / t_x(down) ~ eta_down / eta_up.
# Usage: python -I k_asym.py <results_dir> <ref_tag> <step_tags,comma> [out.json]
import sys, json, numpy as np, pandas as pd

R, ref_tag, tags = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
out = sys.argv[4] if len(sys.argv) > 4 else f"{R}/k_asym.json"
ref = pd.read_csv(f"{R}/{ref_tag}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step"); s0 = int(ref.step.min())
res = {}
for tag in tags:
    try:
        d = pd.read_csv(f"{R}/{tag}_eval.tsv", sep="\t").dropna(subset=["eval_loss"])
    except FileNotFoundError:
        continue
    d = d[(d.step > s0) & (d.step <= ref.step.max())]
    t = (d.step - s0).values; y = d.eval_loss.values - np.interp(d.step, ref.step, ref.eval_loss)
    ys = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values
    plat = float(np.mean(y[(t >= 200) & (t <= 300)]))
    frac = ys / plat
    r = {"lr": float(d.lr.iloc[0]), "plateau": plat}
    for q in (0.5, 0.8):
        hit = np.nonzero(frac >= q)[0]; r[f"t{int(q * 100)}"] = int(t[hit[0]]) if len(hit) else None
    r["dl_at"] = {int(k): float(np.interp(k, t, y)) for k in (5, 10, 20, 50, 100, 300, 1000, 2000)}
    res[tag] = r; print(tag, json.dumps(r), flush=True)
json.dump(res, open(out, "w"), indent=1)
