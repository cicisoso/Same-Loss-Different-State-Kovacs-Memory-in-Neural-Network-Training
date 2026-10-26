# E10 summary table of every Kovacs/reference pair (Supporting Information): matching time, residual cumulative learning
# rate dA (in reference steps) at the match, and the registered statistics of the excursion on set A (peak = max of the
# 3-evaluation running mean over 10-150 steps after release, its time, the signed extremum, area over 0-300 steps, late
# offset over 1000-2000 steps), the gap between the runs at the match on held-out sets A and B, and the relative weight
# norm at the match. A statistic whose window is not covered by the pair is reported as missing, never substituted.
# Every planned pair keeps its row: a pair whose runs have not finished is listed with a status ("pending", or the label
# given as the fourth argument, e.g. "not run" for the final table), and one whose preparation found no match as such.
# Usage: python -I k_table.py <results_dir> <out.json> <out.tex> [missing_label]
import sys, json, numpy as np, pandas as pd

R, out, tex = sys.argv[1], sys.argv[2], sys.argv[3]
MISSING = sys.argv[4] if len(sys.argv) > 4 else "pending"
SEEDS = [(f"kp_sd{i}", f"refL_sd{i}", f"web 30M, seed {i}", "heat-cool", "w") for i in range(1, 6)] + \
        [(f"km_sd{i}", f"refL_sd{i}", f"web 30M, seed {i}", "cool-heat", "w") for i in range(1, 6)]
PAIRS = [("kp", "ref", "web 30M (pilot)", "heat-cool", "e"), ("kpS", "refL", "web 30M, repeat", "heat-cool", "w"),
         ("kp_ta100", "refL", "web 30M", "heat 100 steps", "w"),
         ("kp_ta1000", "refL", "web 30M", "heat 1000 steps", "w"), ("kp_h2", "refL", "web 30M", "heat at 2x", "w"),
         ("kp_l2", "refL", "web 30M", "cool at 1/4", "w"), ("km2", "refL", "web 30M", "cool-heat", "w"),
         ("kp_reset", "refL_reset", "web 30M", "optimizer reset", "e"), ("kp_r1", "refL_r1", "web 30M, unseen data 1", "heat-cool", "w"),
         ("kp_r2", "refL_r2", "web 30M, unseen data 2", "heat-cool", "w")] + SEEDS + \
        [("kpA8", "refA8", "web 30M, from step 8000", "heat-cool", "w"), ("kp_wd0", "refL_wd0", "web 30M, no decay", "heat-cool", "e"),
         ("km_wd0", "refL_wd0", "web 30M, no decay", "cool-heat", "e"), ("big_kp", "big_refL", "web 124M", "heat-cool", "w"),
         ("code_kp", "code_refL", "code 30M", "heat-cool", "e"), ("code_km", "code_refL", "code 30M", "cool-heat", "e"),
         ("pubmed_kp", "pubmed_refL", "PubMed 30M", "heat-cool", "w"), ("pubmed_km", "pubmed_refL", "PubMed 30M", "cool-heat", "e"),
         ("mfn_kp", "mfn_refL", "MovieLens, momentum", "heat-cool", "e"), ("mfn_km", "mfn_refL", "MovieLens, momentum", "cool-heat", "e"),
         ("mf0n_kp", "mf0n_refL", "MovieLens, plain SGD", "heat-cool", "e"), ("mf0n_km", "mf0n_refL", "MovieLens, plain SGD", "cool-heat", "e"),
         ("cifar_kp", "cifar_refL", "CIFAR CNN, momentum", "heat-cool", "e"), ("cifar_km", "cifar_refL", "CIFAR CNN, momentum", "cool-heat", "e"),
         ("cifar0_kp", "cifar0_refL", "CIFAR CNN, plain SGD", "heat-cool", "e"), ("cifar0_km", "cifar0_refL", "CIFAR CNN, plain SGD", "cool-heat", "e")]


# matched runs saved at the match and released by separate continuations of the saved pair (the sham branches of the norm
# interventions, 1500 steps, so their late window is not covered here): matching from the first, excursion from the second
CONT = {"kpS": ("nm_kp_sham", "nm_ref_sham")}


def ld(tag):
    return pd.read_csv(f"{R}/{tag}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")


def stats(t, y):
    ys = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values
    w = (t >= 10) & (t <= 150); a = (t >= 0) & (t <= 300); late = (t >= 1000) & (t <= 2000)
    out = {"peak": None, "t_peak": None, "ext": None, "t_ext": None, "area": None, "late": None, "late_window": None}
    if t.max() >= 150:
        sg = 1.0 if ys[w][np.argmax(np.abs(ys[w]))] > 0 else -1.0
        i = np.nonzero(w)[0][np.argmax(ys[w])]; j = np.nonzero(w)[0][np.argmax(sg * ys[w])]
        out.update(peak=float(ys[i]), t_peak=float(t[i]), ext=float(ys[j]), t_ext=float(t[j]))
    if t.max() >= 300: out["area"] = float(np.trapezoid(y[a], t[a]))
    if t.max() >= 2000 - 60:                       # covered to within one sparse evaluation interval; actual window recorded
        out.update(late=float(y[late].mean()), late_window=[float(t[late].min()), float(t[late].max())])
    return out


rows = []
for kov, ref, system, prot, kind in PAIRS:
    try:
        m, rm = json.load(open(f"{R}/{kov}_meta.json")), json.load(open(f"{R}/{ref}_meta.json"))
        kc, rc = CONT.get(kov, (kov, ref)); k, r = ld(kc), ld(rc)
    except FileNotFoundError:
        rows.append({"kov": kov, "ref": ref, "system": system, "protocol": prot, "registered": kind == "w", "status": MISSING})
        continue
    ph = m["phases"]; mp = [p for p in ph if p.get("matched_at")]
    if not mp:
        rows.append({"kov": kov, "ref": ref, "system": system, "protocol": prot, "registered": kind == "w",
                     "status": "failed preparation (no match)"})
        continue
    TM = mp[0]["matched_at"]; er = rm["phases"][-1]["lr"]
    dA = sum((min(p["end"], TM) - p["start"]) * (p["lr"] / er - 1) for p in ph if p["start"] < TM)
    kk = k[(k.step >= TM) & (k.step <= r.step.max())]; t = (kk.step.values - TM).astype(float)
    y = kk.eval_loss.values - np.interp(kk.step, r.step, r.eval_loss); st = stats(t, y)
    gB = (float(kk.eval_loss_b.values[0] - np.interp(TM, r.step, r.eval_loss_b))
          if "eval_loss_b" in kk and "eval_loss_b" in r and kk.eval_loss_b.notna().all() and r.eval_loss_b.notna().all() else None)
    st.update(gap_A=float(y[0]) if t[0] == 0 else None, gap_B=gB)
    row = {"kov": kov, "ref": ref, "system": system, "protocol": prot, "registered": kind == "w", "status": "ok",
           "match_steps": int(TM - ph[1]["start"]),
           "dA": float(dA), "L_ref": float(np.interp(TM, r.step, r.eval_loss)), **st}
    if "wnorm" in k and "wnorm" in r: row["w_rel"] = float(kk.wnorm.values[0] / np.interp(TM, r.step, r.wnorm) - 1)
    rows.append(row)
    print(kov, {q: (round(v, 4) if isinstance(v, float) else v) for q, v in row.items() if q not in ("kov", "ref", "system", "protocol", "status")})
json.dump(rows, open(out, "w"), indent=1)

f3 = lambda v: "--" if v is None else (f"{v:+.3f}" if abs(v) >= 1e-3 else f"{v:+.4f}").replace("-", "$-$")
mi = lambda x: x.replace("-", "$-$")                                       # typographic minus in every column
num = lambda v, fmt: "--" if v is None else mi(format(v, fmt))
with open(tex, "w") as fh:
    fh.write("\\begin{tabular}{llrrrrrrrrr}\n\\toprule\n"
             "System & Protocol & $t_m$ & $\\Delta A$ & Gap$_A$ & Gap$_B$ & Peak & $t$ & Extremum & Area & Late\\\\\n\\midrule\n")
    for row in rows:
        if row["status"] != "ok":
            fh.write(f"{row['system']}{'' if row['registered'] else '$^{*}$'} & {row['protocol']} & "
                     f"\\multicolumn{{9}}{{l}}{{{row['status']}}}\\\\\n"); continue
        gA = num(None if row["gap_A"] is None else 1e3 * row["gap_A"], "+.2f")
        gB = num(None if row["gap_B"] is None else 1e3 * row["gap_B"], "+.2f")
        fh.write(f"{row['system']}{'' if row['registered'] else '$^{*}$'} & {row['protocol']} & {row['match_steps']} & "
                 f"{num(row['dA'], '.0f')} & {gA} & {gB} & {f3(row['peak'])} & {num(row['t_peak'], '.0f')} & {f3(row['ext'])} & "
                 f"{num(row['area'], '.2f')} & {f3(row['late'])}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print("rows", len(rows), "with results", sum(r["status"] == "ok" for r in rows))
