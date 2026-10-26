# E10 provenance of the replayed preparations of the anneal replications (AN1-AN3): the replays repeat the registered
# heat->cool run of each seed from the same base checkpoint on the same minibatches, but are not bitwise reproducible
# (nondeterministic GPU kernels). For each seed: the match step of the registered run and of the replay, the largest
# difference between replay and registered held-out loss (set A) at common evaluations during heating (300 steps from step
# 4000), their mean difference over the last 100 steps of cooling before the earlier match, and the largest difference
# between the replayed and registered references before the match. Descriptive only.
# Usage: python -I k_replay_check.py <results_dir> [seeds=1,2,3]
import sys, json, pandas as pd

R = sys.argv[1]; seeds = sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "2", "3"]
TM = lambda r: [p for p in json.load(open(f"{R}/{r}_meta.json"))["phases"] if p.get("matched_at")][0]["matched_at"]
load = lambda r: pd.read_csv(f"{R}/{r}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).drop_duplicates("step").set_index("step")["eval_loss"]
for s in seeds:
    try:
        tr, tp = TM(f"kp_sd{s}"), TM(f"kpM_sd{s}"); k, km, r, rm = (load(x) for x in (f"kp_sd{s}", f"kpM_sd{s}", f"refL_sd{s}", f"refM_sd{s}"))
    except (FileNotFoundError, IndexError) as e:
        print(f"seed {s}: pending ({e})"); continue
    d = (km - k).dropna(); t0 = min(tr, tp)
    heat = d[(d.index >= 4000) & (d.index < 4300)]; late = d[(d.index >= t0 - 100) & (d.index < t0)]
    dr = (rm - r).dropna(); dr = dr[(dr.index >= 4000) & (dr.index <= t0)]
    print(f"seed {s}: match registered {tr} replay {tp} (replay - registered {tp - tr:+d}); heating max |dL| {heat.abs().max():.3f} "
          f"(n={len(heat)}); last 100 cooling steps mean dL {late.mean():+.4f} (n={len(late)}); references max |dL| {dr.abs().max():.4f} (n={len(dr)})")
