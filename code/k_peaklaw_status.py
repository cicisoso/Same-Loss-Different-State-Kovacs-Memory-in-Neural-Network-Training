# E10 peak-time law over every planned withheld run (PREREG addendum 05:53), with an explicit status for each run so that no
# denominator shrinks silently (Codex round 4): pending (no meta yet), failed preparation (no match), error (k_peaklaw.py
# raised, e.g. an unbracketed root), undefined prediction, or scored. Scoring itself is k_peaklaw.py, unchanged, called once
# per pair; counts are reported for the law and both baselines against t* and t_max over the scored runs, with the
# geometric-mean factor exp(mean |ln(pred/obs)|). Success is |ln(pred/obs)| < ln 1.5 (strict, as registered); the fitted lag
# t* and the observed maximum t_max are scored separately, each only where it exists and is finite and positive.
# Usage: python -I k_peaklaw_status.py <results_dir> <out.json>
import sys, os, json, subprocess, tempfile, numpy as np

R, out = sys.argv[1], sys.argv[2]; HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = [("kp_ta100", "refL"), ("kp_ta1000", "refL"), ("kp_h2", "refL"), ("kp_l2", "refL"),
        ("kp_sd1", "refL_sd1"), ("kp_sd2", "refL_sd2"), ("kp_sd3", "refL_sd3"), ("kp_sd4", "refL_sd4"), ("kp_sd5", "refL_sd5"),
        ("kp_r1", "refL_r1"), ("kp_r2", "refL_r2"), ("pubmed_kp", "pubmed_refL"), ("big_kp", "big_refL")]
res = {"runs": {}}
for kov, ref in PLAN:
    if not (os.path.exists(f"{R}/{kov}_meta.json") and os.path.exists(f"{R}/{ref}_meta.json")):
        res["runs"][kov] = {"status": "pending"}; continue
    m = json.load(open(f"{R}/{kov}_meta.json"))
    if not [p for p in m["phases"] if p.get("matched_at")]:
        res["runs"][kov] = {"status": "failed preparation (no match)"}; continue
    with tempfile.TemporaryDirectory() as td:
        tmp = f"{td}/pl.json"
        p = subprocess.run([sys.executable, "-I", f"{HERE}/k_peaklaw.py", R, f"{kov}:{ref}", "0", "1", "1", tmp],
                           capture_output=True, text=True)
        try:
            v = json.load(open(tmp))[kov]
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            err = (p.stderr.strip().splitlines() or ["no output"])[-1]
            res["runs"][kov] = {"status": f"error: {err[:200]}"}; continue
    if v.get("pred") is None:
        res["runs"][kov] = {"status": "undefined prediction", **v}; continue
    pos = lambda x: x is not None and np.isfinite(x) and x > 0
    v["t_star_status"] = "ok" if pos(v.get("t_star")) else "two-exponential fit failed"   # each observed lag scored on its own
    v["t_max_status"] = "ok" if pos(v.get("t_max")) else "no observed maximum"
    res["runs"][kov] = {"status": "scored", **v}
sc = {k: v for k, v in res["runs"].items() if v["status"] == "scored"}
res["n_planned"] = len(PLAN); res["n_scored"] = len(sc)
res["n_by_status"] = {s: sum(1 for v in res["runs"].values() if v["status"].split(":")[0] == s)
                      for s in sorted({v["status"].split(":")[0] for v in res["runs"].values()})}
for pk, nm in (("pred", "law"), ("base_const", "constant_lag"), ("base_frac", "constant_fraction")):
    for o in ("t_star", "t_max"):
        val = [v for v in sc.values() if v.get(f"{o}_status") == "ok" and v.get(pk) and v[pk] > 0]
        ok = [abs(np.log(v[pk] / v[o])) < np.log(1.5) for v in val]           # strict, as registered (05:53)
        err = [abs(np.log(v[pk] / v[o])) for v in val]
        res[f"{nm}_{o}"] = {"pass": int(sum(ok)), "n": len(ok), "factor": float(np.exp(np.mean(err))) if err else None}
print(json.dumps({k: v for k, v in res.items() if k != "runs"}))
for k, v in res["runs"].items():
    print(k, v["status"], {q: v.get(q) for q in ("pred", "t_star", "t_max", "ok_t_star", "ok_t_max")} if v["status"] == "scored" else "")
json.dump(res, open(out, "w"), indent=1)
