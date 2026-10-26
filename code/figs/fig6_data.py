# Data for Figure 6 (matched checkpoints are not interchangeable): identical anneals from the two matched checkpoints
# (k_anneal.json), anneal endpoints of independent matched pairs (k_anneal_pairs_*.json), quench probes at the match (kp_probe /
# km_probe against the reference quenched at the same step; dropped from the queue at 10:20), the
# excursion at a later training age (kpA8 against refA8, with the young pair kp against refL), and the excursion split by
# target-token frequency (k_detail_kp.json).
# Usage: python fig6_data.py <results_dir> <out.json> [k_detail json for the token-frequency panels]
import sys, json, numpy as np, pandas as pd

R, out = sys.argv[1], sys.argv[2]; DET = sys.argv[3] if len(sys.argv) > 3 else f"{R}/k_detail_kp.json"
ld = lambda t: pd.read_csv(f"{R}/{t}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
TMF = lambda t: [p for p in json.load(open(f"{R}/{t}_meta.json"))["phases"] if p.get("matched_at")][0]["matched_at"]
res = {}
try:
    res["anneal"] = json.load(open(f"{R}/k_anneal.json"))
except FileNotFoundError:
    res["anneal"] = None
try:                                    # exploratory: anneals after the function-preserving norm changes (k_annealN.py)
    res["annealN"] = json.load(open(f"{R}/k_annealN.json"))
except FileNotFoundError:
    res["annealN"] = None
# the free excursion of the same matched pair (continuations at eta_ref with their own optimizer state), for the
# common-clock prediction of the anneal gap: under a clock shared by both networks the gap at intrinsic time s equals the
# free excursion at s
try:
    fk, fr = ld("sw_kp_own"), ld("sw_ref_own"); t0f = int(fr.step.min())
    res["free"] = {"t": (fk.step.values - t0f).astype(float).tolist(),
                   "d": (fk.eval_loss.values - np.interp(fk.step, fr.step, fr.eval_loss)).tolist(),
                   "d_b": (fk.eval_loss_b.values - np.interp(fk.step, fr.step, fr.eval_loss_b)).tolist()}
except FileNotFoundError:
    res["free"] = None

# independent matched pairs (registered 10:20): anneal endpoints of each pair against its own free excursion
# (k_anneal_pairs.py; the transplant pair against its own-state continuations)
res["pairs"] = {}
for f, keys in ((f"{R}/k_anneal_pairs_tp.json", (("B0sw", "transplant"),)),
                (f"{R}/k_anneal_pairs_sd.json", (("AN1", "seed 1"), ("AN2", "seed 2"), ("AN3", "seed 3")))):
    try:
        J = json.load(open(f))
    except FileNotFoundError:
        continue
    for k, nm in keys:
        if J.get(k, {}).get("status") == "ok": res["pairs"][nm] = J[k]
print("pairs", list(res["pairs"]))

# quench probes: L_kov - L_ref through the 400-step quench, both quenched at the Kovacs match
res["probe"] = {}
for k in ("kp", "km"):
    try:
        a, b, TM = ld(f"{k}_probe"), ld(f"ref_probe_{k}"), TMF(f"{k}_probe")
    except (FileNotFoundError, IndexError):
        continue
    a = a[(a.step >= TM) & (a.step <= b.step.max())]; t = (a.step.values - TM).astype(float)
    res["probe"][k] = {"t": t.tolist(), "d": (a.eval_loss.values - np.interp(a.step, b.step, b.eval_loss)).tolist(),
                       "drop_kov": float(a.eval_loss.values[0] - a.eval_loss.values[-1]),
                       "drop_ref": float(np.interp(TM, b.step, b.eval_loss) - b.eval_loss.values[-1])}
    print("probe", k, {q: round(v, 4) for q, v in res["probe"][k].items() if q.startswith("drop")})

# ageing: the same protocol from a checkpoint 4000 steps older
res["age"] = {}
for nm, rt, kt in (("young", "ref", "kp"), ("old", "refA8", "kpA8")):
    try:
        r, k, TM = ld(rt), ld(kt), TMF(kt)
    except (FileNotFoundError, IndexError):
        continue
    k = k[(k.step >= TM) & (k.step <= r.step.max())]; t = (k.step.values - TM).astype(float)
    m = json.load(open(f"{R}/{kt}_meta.json"))
    res["age"][nm] = {"t": t.tolist(), "d": (k.eval_loss.values - np.interp(k.step, r.step, r.eval_loss)).tolist(),
                      "start": int(m["phases"][0]["start"]), "cool_steps": int(TM - m["phases"][1]["start"])}
    print("age", nm, "start", res["age"][nm]["start"], "cooling", res["age"][nm]["cool_steps"])

# token-frequency decomposition of the matched pair
try:
    dk = json.load(open(DET)); A = dk["A"]
    res["fbin"] = {q: A[q] for q in ("t", "fbin_edges", "fbin_weight", "fbin_curves", "fbin_at_match") if q in A}
except (FileNotFoundError, KeyError):
    res["fbin"] = None
json.dump(res, open(out, "w"))
print("anneal", res["anneal"] is not None, "probe", list(res["probe"]), "age", list(res["age"]), "fbin", res["fbin"] is not None)
