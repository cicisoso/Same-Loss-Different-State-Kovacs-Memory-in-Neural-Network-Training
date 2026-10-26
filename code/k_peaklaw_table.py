# E10 Supporting-Information table of the peak-time law over every planned withheld run (k_peaklaw_status.py output): the
# protocol, the measured cooling time (steps and intrinsic), the lag predicted by the law and by the two baselines, the
# observed lags (two-exponential fit t* and smoothed maximum t_max), with a mark for each prediction outside the factor 1.5;
# runs without a score are listed with their status.
# Usage: python -I k_peaklaw_table.py <k_peaklaw_status.json> <out.tex>
import sys, json, numpy as np

D = json.load(open(sys.argv[1])); out = sys.argv[2]
NAMES = {"kp_ta100": ("web 30M", "heat 100 steps"), "kp_ta1000": ("web 30M", "heat 1000 steps"),
         "kp_h2": ("web 30M", r"heat at $2\eref$"), "kp_l2": ("web 30M", r"cool at $\eref/4$"),
         **{f"kp_sd{s}": (f"web 30M, seed {s}", "heat--cool") for s in range(1, 6)},
         "kp_r1": ("web 30M, unseen data 1", "heat--cool"), "kp_r2": ("web 30M, unseen data 2", "heat--cool"),
         "pubmed_kp": ("PubMed 30M", "heat--cool"), "big_kp": ("web 124M", "heat--cool")}
ok = lambda p, o: bool(p and o and p > 0 and o > 0 and abs(np.log(p / o)) < np.log(1.5))   # strict, as registered
def cell(p, t, m):   # marks count only the observed lags that exist (a failed fit is not a miss)
    if p is None: return "--"
    miss = [o for o in (t, m) if o is not None and not ok(p, o)]
    return f"{p:.1f}" + ("" if not miss else r"$^{\dagger}$" if len(miss) == 2 else r"$^{*}$")
lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Network & Protocol & $t_c$ & $s_c$ & Law & Lag & Fraction & $t^*$ & $t_{\max}$\\", r"\midrule"]
for k, v in D["runs"].items():
    net, prot = NAMES.get(k, (k, ""))
    if v["status"] != "scored":
        lines.append(f"{net} & {prot} & \\multicolumn{{7}}{{l}}{{{v['status'].split(':')[0]}}}\\\\"); continue
    t, m = v.get("t_star"), v.get("t_max")
    lines.append(f"{net} & {prot} & {v['t_c']} & {v['s_c']:.1f} & {cell(v['pred'], t, m)} & {cell(v['base_const'], t, m)} & "
                 f"{cell(v['base_frac'], t, m)} & {'--' if t is None else f'{t:.1f}'} & {'--' if m is None else f'{m:.0f}'}\\\\")
lines += [r"\midrule"]
for nm, lab in (("law", "Law"), ("constant_lag", "Constant lag"), ("constant_fraction", "Constant fraction")):
    a, b = D.get(f"{nm}_t_star", {}), D.get(f"{nm}_t_max", {})
    if a and b:
        lines.append(f"\\multicolumn{{4}}{{l}}{{{lab}: within 1.5 of $t^*$ / $t_{{\\max}}$}} & \\multicolumn{{5}}{{l}}{{{a['pass']} of "
                     f"{a['n']} / {b['pass']} of {b['n']}; typical factor {a['factor']:.2f} / {b['factor']:.2f}}}\\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
