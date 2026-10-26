# E10 Supporting-Information table of the registered contest (PREREG.md; scored by k_score.py, unchanged): for every scored
# Kovacs run and each of the six frozen fitted implementations, the sign of the observed and predicted maximum, the
# predicted-to-observed height and lag ratios of the maximum (three-evaluation running mean, 10-150 steps), the NRMSE of the
# post-release excursion (normalised by the range of the smoothed observed excursion), the RMSE of the absolute loss over
# the whole run, and the registered verdict (pass: correct sign, both ratios within a factor 1.5, NRMSE < 0.3). The
# exploratory first network is listed first and is not counted. Also writes the counts per implementation to <out.json>.
# The output is a complete longtable; si.tex defines \contestcaption and inputs it outside any float.
# Usage: python -I k_contest_table.py <out.tex> <out.json> <label=models.json,...>   (each models.json has a *_score.json)
import sys, json, os

out_tex, out_json, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
MODELS = [("tissue", "single-timescale"), ("mpl", "multi-power"), ("fsl", "functional scaling"), ("nqm2", "two modes"),
          ("nqm2pq", "two modes, free $p,q$"), ("nqmS", "26 modes")]
LABEL = {"kp": "pilot (exploratory)", "kp_ta100": "heat 100 steps", "kp_ta1000": "heat 1000 steps", "kp_h2": "heat at $2\\eta_{\\mathrm{ref}}$",
         "kp_l2": "cool at $\\eta_{\\mathrm{ref}}/4$", "km2": "cool-heat", "kp_r1": "unseen data 1", "kp_r2": "unseen data 2",
         "kpA8": "from step 8000", **{f"kp_sd{i}": f"seed {i}" for i in range(1, 6)},
         **{f"km_sd{i}": f"seed {i}, cool-heat" for i in range(1, 6)}}
mi = lambda x: x.replace("-", "$-$")
# a longtable (the table runs over more than one page); its caption text is the macro \contestcaption defined in si.tex
HEAD = [r"\toprule", r"Run & Implementation & Sign obs./pred. & Height & Lag & NRMSE & RMSE$_{\mathrm{abs}}$ & Pass\\", r"\midrule"]
lines = [r"\begin{longtable}{llcrrrrc}", r"\caption{\contestcaption}\label{tab:contest}\\", *HEAD, r"\endfirsthead",
         r"\multicolumn{8}{l}{Table~\thetable{} (continued)}\\", *HEAD, r"\endhead", r"\bottomrule", r"\endlastfoot"]
res = {"runs": [], "counts": {m: {"n": 0, "pass": 0} for m, _ in MODELS}}
first = True
for spec in specs:
    kind, f = spec.split("=")
    sf = f.replace(".json", "_score.json")
    if not (os.path.exists(f) and os.path.exists(sf)): continue
    M, S = json.load(open(f)), json.load(open(sf))
    for run, row in S.items():
        if run.startswith("_") or not row: continue
        tag = run.split(":")[0]
        if not first: lines.append(r"\addlinespace")
        first = False
        for i, (m, name) in enumerate(MODELS):
            if m not in row: continue
            v = row[m]; rm = M.get(run, {}).get(m, {}).get("rmse_abs")
            sg = lambda x: "+" if x > 0 else "$-$" if x < 0 else "0"
            cells = [LABEL.get(tag, tag) if i == 0 else "", name, f"{sg(v['obs']['peak'])}/{sg(v['pred']['peak'])}",
                     mi(f"{v['height_ratio']:.2f}"), mi(f"{v['time_ratio']:.2f}"), f"{v['nrmse']:.2f}",
                     "--" if rm is None else f"{rm:.3f}", "yes" if v["success"] else "no"]
            lines.append(" & ".join(cells) + r"\\")
            res["runs"].append({"run": run, "kind": kind, "model": m, "height_ratio": v["height_ratio"], "time_ratio": v["time_ratio"],
                                "nrmse": v["nrmse"], "rmse_abs": rm, "pass": v["success"]})
            if kind != "exploratory":
                res["counts"][m]["n"] += 1; res["counts"][m]["pass"] += int(v["success"])
lines += [r"\end{longtable}"]
open(out_tex, "w").write("\n".join(lines) + "\n"); json.dump(res, open(out_json, "w"), indent=1)
print({m: f"{c['pass']}/{c['n']}" for m, c in res["counts"].items()})
