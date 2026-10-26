# Data for Figure 3 (where the memory lives): transplant factorial (k_swap.json), norm interventions (k_norm.json), and the
# norms of the weight-matrix groups through the protocol, Kovacs (kpS to the match, then sw_kp_own) against the reference
# (refL), from the per-matrix norms in the *_detail.npz files; angular updates per group from the *_angle.tsv files.
# Usage: python fig3_data.py <results_dir> <out.json>
import sys, json, numpy as np, pandas as pd

R, out = sys.argv[1], sys.argv[2]
res = {}
for key, f in (("swap", "k_swap.json"), ("norm", "k_norm.json")):
    try:
        res[key] = json.load(open(f"{R}/{f}"))
    except FileNotFoundError:
        res[key] = None

GROUPS = {"embedding": ("wte.", "wpe."), "attention in": ("qkv.",), "attention out": ("proj.",), "MLP in": ("fc.",),
          "MLP out": ("out.",)}


def group_norms(tag):
    d = np.load(f"{R}/{tag}_detail.npz", allow_pickle=True); names = [str(x) for x in d["lnorm_names"]]
    L = np.asarray(d["lnorm"], float); out_ = {}
    for g, keys in GROUPS.items():
        idx = [i for i, n in enumerate(names) if any(k in n for k in keys)]
        out_[g] = np.sqrt((L[:, idx] ** 2).sum(1))
    return np.asarray(d["step"], float), out_


try:
    sk1, nk1 = group_norms("kpS"); sk2, nk2 = group_norms("sw_kp_own"); sr, nr = group_norms("refL")
    sk = np.r_[sk1, sk2[sk2 > sk1.max()]]; nk = {g: np.r_[nk1[g], nk2[g][sk2 > sk1.max()]] for g in GROUPS}
    keep = sk <= sr.max()
    meta = json.load(open(f"{R}/kpS_meta.json")); TM = [p for p in meta["phases"] if p.get("matched_at")][0]["matched_at"]
    res["norms"] = {"t0": float(sr.min()), "match": float(TM), "heat_end": float(meta["phases"][0]["end"]),
                    "step": (sk[keep] - sr.min()).tolist(),
                    "ratio": {g: (nk[g][keep] / np.interp(sk[keep], sr, nr[g]) - 1).tolist() for g in GROUPS}}
except FileNotFoundError as e:
    res["norms"] = None; print("norm traces missing:", e.filename)

ang = {}
for tag in ("nm_kp_sham", "nm_ref_sham", "nm_kp_proj", "nm_kp_resid", "nm_kp_both", "nm_ref_proj", "nm_ref_resid"):
    try:
        a = pd.read_csv(f"{R}/{tag}_angle.tsv", sep="\t")
    except FileNotFoundError:
        continue
    ang[tag] = {"step": a.step.tolist(), **{c[4:]: a[c].tolist() for c in a.columns if c.startswith("ang_")}}
res["angle"] = ang
json.dump(res, open(out, "w"))
print("swap", res["swap"] is not None, "norm", res["norm"] is not None, "norm traces", res["norms"] is not None, "angles", list(ang))
