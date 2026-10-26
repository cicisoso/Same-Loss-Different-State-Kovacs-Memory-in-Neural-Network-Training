# E10 Supporting-Information table of the anneal replications on independent networks (PREREG addendum 10:20, counting
# note 11:49): for each seed, the replay provenance (match step of the registered Kovacs run and of its replay, gap of the
# replayed pair when the anneal starts, sets A and B), the final anneal gap G_s at the last evaluation and as the mean of the
# last three, each with its block-bootstrap 95% interval on sets A and B, the two secondary contrasts (G_s - shared clock and
# G_s - largest free excursion of the registered pair) with their joint intervals, and the outcome of the registered
# criterion on set A at the last evaluation: met (interval above zero), unresolved (interval crosses zero), opposite
# (interval below zero) or unscored. Also writes the counts to <out.json>.
# Usage: python -I k_anneal_rep_table.py <results_dir> <k_anneal_pairs_sd.json> <out.tex> <out.json>
import sys, json, os

R, f, out_tex, out_json = sys.argv[1:5]
D = json.load(open(f)) if os.path.exists(f) else {}
mi = lambda x: x.replace("-", "$-$")
num = lambda v, d=3: "--" if v is None else mi(f"{v:.{d}f}").replace("$-$0." + "0" * d, "0." + "0" * d)
ci = lambda c: "" if not c else " (" + mi(f"{c[0]:.3f}") + ", " + mi(f"{c[1]:.3f}") + ")"


def matched_at(tag):
    try:
        return [p for p in json.load(open(f"{R}/{tag}_meta.json"))["phases"] if p.get("matched_at")][0]["matched_at"]
    except (FileNotFoundError, IndexError, KeyError):
        return None


def outcome(c):
    if not c: return "unscored"
    return "met" if c[0] > 0 else "opposite" if c[1] < 0 else "unresolved"


# Two panels (each fits the page at the table's font size): a) provenance, final gaps and outcome; b) secondary contrasts.
LA = [r"\multicolumn{8}{l}{a) Final gaps}\\", r"\toprule",
      r"Seed & Set & Match & Replay & Start & $G_s$ last & $G_s$ last three & Outcome\\", r"\midrule"]
LB = [r"\multicolumn{4}{l}{b) Secondary contrasts, with joint block-bootstrap 95\% intervals}\\", r"\toprule",
      r"Seed & Set & $G_s-$clock & $G_s-$max.\\", r"\midrule"]
res = {"pairs": {}}
for s in (1, 2, 3):
    e = D.get(f"AN{s}", {})
    t_reg, t_rep = matched_at(f"kp_sd{s}"), matched_at(f"kpM_sd{s}")
    if e.get("status") != "ok":
        st = e.get("status", "pending").split(":")[0]
        st = "pending" if st.startswith("missing") else st              # an input file not yet written: the run is pending
        res["pairs"][s] = {"outcome": "unscored", "status": st, "match_registered": t_reg, "match_replay": t_rep}
        LA.append(f"{s} & & {t_reg or '--'} & {t_rep or '--'} & \\multicolumn{{4}}{{l}}{{{st}}}\\\\")
        LB.append(f"{s} & & \\multicolumn{{2}}{{l}}{{{st}}}\\\\"); continue
    r = {"match_registered": t_reg, "match_replay": t_rep}
    for S in ("A", "B"):
        g = lambda k: e.get(f"{S}_{k}")
        r[S] = {"start": g("pre_anneal"), "G_last": g("final_last"), "G_last_ci": g("final_last_ci"),
                "G_last3": g("final_last3"), "G_last3_ci": g("final_last3_ci"),
                "D_clock": g("final_last_minus_clock"), "D_clock_ci": g("final_last_minus_clock_ci"),
                "D_max": g("final_last_minus_fmax"), "D_max_ci": g("final_last_minus_fmax_ci"),
                "outcome": outcome(g("final_last_ci")), "outcome_last3": outcome(g("final_last3_ci")),
                "clock": outcome(g("final_last_minus_clock_ci")), "max": outcome(g("final_last_minus_fmax_ci"))}
        LA.append(f"{s if S == 'A' else ''} & {S} & {(t_reg or '--') if S == 'A' else ''} & {(t_rep or '--') if S == 'A' else ''} & "
                  f"{num(r[S]['start'], 4)} & {num(r[S]['G_last'])}{ci(r[S]['G_last_ci'])} & "
                  f"{num(r[S]['G_last3'])}{ci(r[S]['G_last3_ci'])} & {r[S]['outcome'] if S == 'A' else '(' + r[S]['outcome'] + ')'}\\\\")
        LB.append(f"{s if S == 'A' else ''} & {S} & {num(r[S]['D_clock'])}{ci(r[S]['D_clock_ci'])} & "
                  f"{num(r[S]['D_max'])}{ci(r[S]['D_max_ci'])}\\\\")
    r["outcome"] = r["A"]["outcome"]                                   # registered: set A, last evaluation (note 11:49)
    r["robust_agree"] = all(x == r["outcome"] for x in (r["B"]["outcome"], r["A"]["outcome_last3"], r["B"]["outcome_last3"]))
    res["pairs"][s] = r
lines = ([r"\begin{tabular}{llrrrllc}"] + LA + [r"\bottomrule", r"\end{tabular}", "", r"\vspace{8pt}", "",
         r"\begin{tabular}{llll}"] + LB + [r"\bottomrule", r"\end{tabular}"])
P = res["pairs"].values()
res["n_met"] = sum(p["outcome"] == "met" for p in P); res["n_scored"] = sum(p["outcome"] != "unscored" for p in P)
res["n_opposite"] = sum(p["outcome"] == "opposite" for p in P); res["n_unresolved"] = sum(p["outcome"] == "unresolved" for p in P)
res["n_clock_above"] = sum(p.get("A", {}).get("clock") == "met" for p in P)
res["n_max_above"] = sum(p.get("A", {}).get("max") == "met" for p in P)
open(out_tex, "w").write("\n".join(lines) + "\n"); json.dump(res, open(out_json, "w"), indent=1)
print("\n".join(lines)); print({k: v for k, v in res.items() if k != "pairs"})
