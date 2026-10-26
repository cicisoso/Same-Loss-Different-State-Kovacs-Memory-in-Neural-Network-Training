# Data for Figure 1 (phenomenon): absolute losses and learning rates of the flagship pair, and post-release excursions of
# every available Kovacs/reference pair (set A, set B, per-step training loss on the shared minibatches).
# Usage: python fig1_data.py <results_dir> <out.json> <ref:kov,ref:kov,...>   (first pair = the one drawn in full; pairs whose
# Kovacs tag is not kp_sd* are exploratory and appear only as grey lines in panel e)
import sys, json, numpy as np, pandas as pd

R, out, pairs = sys.argv[1], sys.argv[2], [p.split(":") for p in sys.argv[3].split(",")]
ld = lambda t: pd.read_csv(f"{R}/{t}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
res = {"pairs": []}
for i, (rt, kt) in enumerate(pairs):
    try:
        r, k, m = ld(rt), ld(kt), json.load(open(f"{R}/{kt}_meta.json"))
    except FileNotFoundError:
        continue
    TM = [p for p in m["phases"] if p.get("matched_at")][0]["matched_at"]
    k = k[(k.step >= r.step.min()) & (k.step <= r.step.max())]
    e = {"ref": rt, "kov": kt, "TM": int(TM), "phases": [[p["start"], p["end"], p["lr"]] for p in m["phases"]],
         "seed": kt.startswith("kp_sd")}                              # registered independent networks; others drawn in grey
    t = (k.step - TM).values; e["t"] = t.tolist()
    e["dA"] = (k.eval_loss.values - np.interp(k.step, r.step, r.eval_loss)).tolist()
    if "eval_loss_b" in k and "eval_loss_b" in r and not r.eval_loss_b.isna().all():
        e["dB"] = (k.eval_loss_b.values - np.interp(k.step, r.step, r.eval_loss_b)).tolist()
    try:                                                              # per-step training loss on identical minibatches
        tr, tk = (pd.read_csv(f"{R}/{x}_train.tsv", sep="\t").set_index("step").loss for x in (rt, kt))
        common = tk.index.intersection(tr.index); d = (tk[common] - tr[common]).sort_index()
        e["train_t"] = (d.index.values - TM).tolist(); e["dTrain"] = d.values.tolist()
    except (FileNotFoundError, KeyError):
        pass
    if i == 0:
        e["abs"] = {"ref_step": r.step.tolist(), "ref_loss": r.eval_loss.tolist(), "kov_step": k.step.tolist(),
                    "kov_loss": k.eval_loss.tolist()}
    res["pairs"].append(e); print(kt, "TM", TM, "n", len(t), "B" if "dB" in e else "-", "train" if "dTrain" in e else "-")
json.dump(res, open(out, "w"))
