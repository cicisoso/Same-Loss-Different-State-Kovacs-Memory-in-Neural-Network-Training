# Early minimum of the cool-heat excursions after release (unsmoothed, set A) within 20, 50 and 100 steps, and the last
# step within 100 steps at which the excursion is negative; web text (exploratory checkpoint, five networks, no weight
# decay), code and PubMed. Usage: python -I kmdip.py <results_dir>
import sys, json, numpy as np, pandas as pd
R = sys.argv[1]
for kov, ref in [("km2", "refL")] + [(f"km_sd{s}", f"refL_sd{s}") for s in range(1, 6)] + [("km_wd0", "refL_wd0"), ("code_km", "code_refL"), ("pubmed_km", "pubmed_refL")]:
    m = json.load(open(f"{R}/{kov}_meta.json")); TM = [p for p in m["phases"] if p.get("matched_at")][0]["matched_at"]
    k, r = (pd.read_csv(f"{R}/{x}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step") for x in (kov, ref))
    out = []
    for hi in (20, 50, 100):
        kk = k[(k.step > TM) & (k.step <= TM + hi)]; y = kk.eval_loss.values - np.interp(kk.step, r.step, r.eval_loss)
        out.append(f"1-{hi}: {y.min():+.4f} at {kk.step.values[np.argmin(y)] - TM}")
    kk = k[(k.step > TM) & (k.step <= TM + 100)]; y = kk.eval_loss.values - np.interp(kk.step, r.step, r.eval_loss)
    neg = (kk.step.values - TM)[y < 0]
    print(kov, "; ".join(out), "; last negative step within 100:", neg.max() if len(neg) else None)
