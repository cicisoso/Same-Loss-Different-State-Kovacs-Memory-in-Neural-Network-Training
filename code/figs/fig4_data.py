# Data for Figure 4 (prediction): per-unit step responses in intrinsic time, blind contest curves for the Kovacs runs, the
# pre-registered scores, the peak-time-law table and the inferred relaxation spectrum. Reads only outputs of the analysis
# scripts (k_lin.py, k_models.py, k_score.py, k_peaklaw.py).
# The step responses are taken river-corrected from k_super.py output when its JSON is given in place of k_lin.json.
# Usage: python fig4_data.py <out.json> <k_lin.json|k_super.json> <contest.json,contest.json,...> <peaklaw.json,peaklaw.json,...>
# (a k_peaklaw_status.json among the peak-law files contributes its scored runs)
import sys, json, numpy as np

out, lin_f, contest_fs, peak_fs = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4].split(",")
L = json.load(open(lin_f))
res = {"steps": {k: {"eta": v["eta"], "t": v["t"], "per_unit": v["per_unit"]} for k, v in L["steps"].items()},
       "runs": {}, "scores": {}, "spectrum": None, "peaklaw": {}, "river_corrected": "R" in L}
if "R" in L:                                                      # logarithmic reference line from the cooling steps
    s, Rm = np.array(L["R"]["s"]), np.array(L["R"]["R"], float); m = np.isfinite(Rm) & (s >= 1) & (s <= 50)
    w, c = np.polyfit(np.log(s[m]), Rm[m], 1); res["logline"] = {"w": float(w), "c": float(c), "s": [1.0, 50.0]}
for f in contest_fs:
    C = json.load(open(f))
    if res["spectrum"] is None and "nqmS_spectrum" in C:
        res["spectrum"] = C["nqmS_spectrum"]; res["pq"] = C.get("nqmS_pq")
    try:
        S = json.load(open(f.replace(".json", "_score.json")))
    except FileNotFoundError:
        S = {}
    for run, rec in C.items():
        if not isinstance(rec, dict) or "matched_at" not in rec: continue
        res["runs"][run] = {m: rec[m]["curve"] for m in ("tissue", "mpl", "fsl", "nqm2", "nqmS") if m in rec and "curve" in rec[m]}
        if run in S: res["scores"][run] = S[run]
for f in peak_fs:                                                 # k_peaklaw_status.json: its scored runs, given last, win
    J = json.load(open(f))
    res["peaklaw"].update({k: v for k, v in J["runs"].items() if v["status"] == "scored"} if "runs" in J else J)
json.dump(res, open(out, "w"))
print("runs", list(res["runs"]), "peaklaw", list(res["peaklaw"]))
