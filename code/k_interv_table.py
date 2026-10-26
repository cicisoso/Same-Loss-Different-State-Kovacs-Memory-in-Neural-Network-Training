# E10 Supporting-Information table of the interventions at the match: the registered statistics of every norm branch
# (k_norm.json: peak, area, late offset over 1000-1500 steps, against the unchanged reference continuation; Kovacs branches
# also as differences from the unchanged Kovacs continuation) and of the optimizer-state transplant (k_swap.json: observed
# excursion, simple weight and state effects and their interaction, which add up to the excursion). Missing windows are
# shown as dashes.
# Usage: python -I k_interv_table.py <k_norm.json> <k_swap.json> <out.tex> [k_pairs_rst.json] [k_pairs_reset.json]
#   the optional files add the optimizer resets (PREREG note 10:36): both networks' AdamW moments cleared at the match of the
#   transplant pair (rs_kp - rs_ref against sw_kp_own - sw_ref_own) and at the release of a repeated preparation from the
#   base checkpoint (kp_reset - refL_reset against kp - refL), each from the Kovacs run's match
import sys, json, os

kn = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {"branches": {}}
ks = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {}
mi = lambda x: x.replace("-", "$-$")
num = lambda v, fmt: "--" if v is None else mi(format(v, fmt))
B = kn.get("branches", {}); S = lambda t, k: B.get(t, {}).get("stats_A", {}).get(k)
NAMES = {"nm_kp_sham": ("Kovacs", "unchanged"), "nm_kp_proj": ("Kovacs", "attn./MLP inputs reduced"),
         "nm_kp_resid": ("Kovacs", "residual stream reduced"), "nm_kp_both": ("Kovacs", "both reduced"),
         "nm_ref_proj": ("reference", "attn./MLP inputs increased"), "nm_ref_resid": ("reference", "residual stream increased")}
lines = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
         "Network & Change at the match & $\\delta_0$ & Peak & $t$ & Area & Late & $\\Delta$Peak & $\\Delta$Area\\\\", "\\midrule"]
for tag, (net, ch) in NAMES.items():
    if tag not in B: continue
    d0 = B[tag].get("first_eval_diff"); kov = tag.startswith("nm_kp") and tag != "nm_kp_sham"
    dp = None if not kov or S(tag, "peak") is None or S("nm_kp_sham", "peak") is None else S(tag, "peak") - S("nm_kp_sham", "peak")
    da = None if not kov or S(tag, "area") is None or S("nm_kp_sham", "area") is None else S(tag, "area") - S("nm_kp_sham", "area")
    pk, tp = (S(tag, "peak"), S(tag, "t_peak")) if net == "Kovacs" else (None, None)   # registered for the reference: area, late
    lines.append(f"{net} & {ch} & {num(None if d0 is None else 1e4 * d0, '+.1f')} & {num(pk, '+.4f')} & "
                 f"{num(tp, '.0f')} & {num(S(tag, 'area'), '+.2f')} & {num(S(tag, 'late'), '+.4f')} & "
                 f"{num(dp, '+.4f')} & {num(da, '+.2f')}\\\\")
lines += ["\\midrule", "\\multicolumn{9}{l}{Optimizer-state transplant (set A)}\\\\",
          "Weights & Optimizer state & & Peak & $t$ & Area & Late & & \\\\", "\\midrule"]
st = ks.get("eval_loss", {}).get("stats", {})
for key, (w, o) in (("own", ("Kovacs", "Kovacs")), ("weights", ("Kovacs", "reference")), ("state", ("reference", "Kovacs")),
                    ("I", ("interaction", ""))):
    if key not in st: continue
    v = dict(st[key]); fz = lambda x: None if x is None else (0.0 if abs(x) < 5e-5 else x)      # no "+0.0000"
    if key == "I": v["peak"] = v["t_peak"] = None          # maxima do not decompose additively: no maximum for the interaction
    lines.append(f"{w} & {o} & & {num(v.get('peak'), '+.4f')} & {num(v.get('t_peak'), '.0f')} & {num(v.get('area'), '+.2f')} & "
                 f"{num(fz(v.get('late')), '+.4f').replace('+0.0000', '0.0000')} & & \\\\")
opt = lambda i: json.load(open(sys.argv[i])) if len(sys.argv) > i and os.path.exists(sys.argv[i]) else {}
RS = [opt(4), opt(5)]
blocks = [(RS[0], "Transplant pair", "rs_kp:rs_ref", "sw_kp_own:sw_ref_own"), (RS[1], "From step 4000", "kp_reset:refL_reset", "kp:refL")]
rows = []
for rs, nm, kr, ku in blocks:
    kr_ = next((k for k in rs if k.startswith(kr + "@")), None); ku_ = next((k for k in rs if k.startswith(ku + "@")), None)
    if not kr_ or not ku_: continue
    a, b = rs[kr_].get("stats_A", {}), rs[ku_].get("stats_A", {})
    for lab, v, dd in (("kept", b, False), ("both reset", a, True)):
        dp = a["peak"] - b["peak"] if dd and a.get("peak") is not None and b.get("peak") is not None else None
        da = a["area"] - b["area"] if dd and a.get("area") is not None and b.get("area") is not None else None
        rows.append(f"{nm if not dd else ''} & {lab} & & {num(v.get('peak'), '+.4f')} & {num(v.get('t_peak'), '.0f')} & "
                    f"{num(v.get('area'), '+.2f')} & {num(v.get('late'), '+.4f')} & {num(dp, '+.4f')} & {num(da, '+.2f')}\\\\")
if rows:
    lines += ["\\midrule", "\\multicolumn{9}{l}{Optimizer reset at the match (set A)}\\\\",
              "Pair & Optimizer states & & Peak & $t$ & Area & Late & $\\Delta$Peak & $\\Delta$Area\\\\", "\\midrule"] + rows
lines += ["\\bottomrule", "\\end{tabular}"]
open(sys.argv[3], "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
