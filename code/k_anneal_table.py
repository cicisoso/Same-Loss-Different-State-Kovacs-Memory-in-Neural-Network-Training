# E10 Supporting-Information table of the anneal endpoints: for every annealed matched pair (k_anneal_pairs.py output) the
# gap when the decay starts, the final gap (last evaluation), the largest free excursion of the same pair, the shared-clock
# prediction (free excursion at intrinsic time B + 250.5), the two registered contrasts with their joint block-bootstrap 95%
# intervals, on sets A and B; and, on set A, the range of the final gaps predicted by the fitted schedule-to-loss models
# (k_models.py, "final_pred" of the pair). Rows whose runs are missing or incomplete are listed as pending.
# Usage: python -I k_anneal_table.py <out.tex> <models1.json,models2.json,...> "<label>|<pairs.json>|<key>" ...
import sys, json, os

out, mfiles, rows = sys.argv[1], [f for f in sys.argv[2].split(",") if f], sys.argv[3:]
MODELS = {}
for f in mfiles:
    if os.path.exists(f):
        MODELS.update({k: v for k, v in json.load(open(f)).items() if ":" in k and isinstance(v, dict)})
mi = lambda x: x.replace("-", "$-$")
num = lambda v, d=3: "--" if v is None else mi(f"{v:.{d}f}").replace("$-$0." + "0" * d, "0." + "0" * d)   # no "-0.000"
ci = lambda c: "" if not c else " (" + mi(f"{c[0]:.3f}") + ", " + mi(f"{c[1]:.3f}") + ")"
# Two panels (each fits the page at the table's font size): a) endpoints; b) the registered contrasts with their intervals.
LA = ["\\multicolumn{8}{l}{a) Endpoints}\\\\", "\\toprule",
      "Pair & $t_{\\mathrm{d}}$ & Set & Start & Final & Free max. & Clock & Models\\\\", "\\midrule"]
LB = ["\\multicolumn{5}{l}{b) Contrasts, with joint block-bootstrap 95\\% intervals}\\\\", "\\toprule",
      "Pair & $t_{\\mathrm{d}}$ & Set & Final$-$max. & Final$-$clock\\\\", "\\midrule"]
cache = {}
for r in rows:
    label, f, key = r.split("|")
    if f not in cache: cache[f] = json.load(open(f)) if os.path.exists(f) else {}
    e = cache[f].get(key)
    if not e or e.get("status") != "ok":
        LA.append(f"{label} & & & \\multicolumn{{5}}{{l}}{{pending}}\\\\"); LB.append(f"{label} & & & \\multicolumn{{2}}{{l}}{{pending}}\\\\")
        continue
    mk = MODELS.get(f"{e['pair'][0]}:{e['pair'][1]}", {})
    fp = [v["final_pred"] for k, v in mk.items() if isinstance(v, dict) and "final_pred" in v]
    mr = "--" if not fp else mi(f"{min(fp):.4f}") + ", " + mi(f"{max(fp):.4f}")
    for S in ("A", "B"):
        g = lambda k: e.get(f"{S}_{k}")
        head = f"{label if S == 'A' else ''} & {e['B'] if S == 'A' else ''} & {S}"
        LA.append(f"{head} & {num(g('pre_anneal'), 4)} & {num(g('final_last'))} & {num(g('fmax'))} & {num(g('clock'))} & "
                  f"{mr if S == 'A' else ''}\\\\")
        LB.append(f"{head} & {num(g('final_last_minus_fmax'))}{ci(g('final_last_minus_fmax_ci'))} & "
                  f"{num(g('final_last_minus_clock'))}{ci(g('final_last_minus_clock_ci'))}\\\\")
lines = (["\\begin{tabular}{lrlrrrrr}"] + LA + ["\\bottomrule", "\\end{tabular}", "", "\\vspace{8pt}", "",
         "\\begin{tabular}{lrlll}"] + LB + ["\\bottomrule", "\\end{tabular}"])
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
