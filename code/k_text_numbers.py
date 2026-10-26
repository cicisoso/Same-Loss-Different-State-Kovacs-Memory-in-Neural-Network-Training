# E10 the numbers quoted in the manuscript text, read from the analysis outputs only (no new statistics): the registered
# maximum of the five independent networks on sets A and B (mean +- s.e.m. over the seeds whose pair is complete; set B with
# the same statistic as k_table.py), the peak-time-law counts and typical factors (k_peaklaw_status.json), the registered
# contest (k_contest.json: passes per implementation, component counts, ranges of the 26-mode ratios on the withheld runs), the
# token-bin factors (k_detail_<run>.json),
# the excursion statistics of the breadth, replicate, ageing and reverse runs (k_table.json on set A, recomputed on set B),
# the anneal replications (k_anneal_rep.json) and the reset classifications (k_reset_rule.json).
# Usage: python -I k_text_numbers.py <results_dir>
import sys, json, os, numpy as np, pandas as pd

R = sys.argv[1]
J = lambda f: json.load(open(f"{R}/{f}")) if os.path.exists(f"{R}/{f}") else None


def peak(t, y):                                                    # k_table.py: max of the 3-evaluation running mean, 10-150
    ys = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values; w = (t >= 10) & (t <= 150)
    i = np.nonzero(w)[0][np.argmax(ys[w])]; return float(ys[i]), float(t[i])


print("== independent networks (registered maximum, nats)")
pk = {"A": [], "B": []}
for s in range(1, 6):
    kov, ref = f"kp_sd{s}", f"refL_sd{s}"
    try:
        m = json.load(open(f"{R}/{kov}_meta.json"))
        k = pd.read_csv(f"{R}/{kov}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
        r = pd.read_csv(f"{R}/{ref}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
    except FileNotFoundError:
        print(f"seed {s}: pending"); continue
    TM = [p for p in m["phases"] if p.get("matched_at")][0]["matched_at"]
    kk = k[(k.step >= TM) & (k.step <= r.step.max())]; t = (kk.step.values - TM).astype(float)
    if t.max() < 150: print(f"seed {s}: incomplete"); continue
    out = []
    for S, col in (("A", "eval_loss"), ("B", "eval_loss_b")):
        y = kk[col].values - np.interp(kk.step, r.step, r[col]); h, tp = peak(t, y); pk[S].append(h); out.append(f"{S} {h:+.4f} at {tp:g}")
    print(f"seed {s}: match {TM}; " + "; ".join(out))
for S in ("A", "B"):
    v = np.array(pk[S])
    if len(v) > 1: print(f"set {S}: n={len(v)} mean {v.mean():.4f} s.e.m. {v.std(ddof=1) / np.sqrt(len(v)):.4f} range {v.min():.4f}-{v.max():.4f}")
try:   # Figure 1 network (seed 1): training loss on the shared minibatches, centered nine-step running mean (as in Figure 1d)
    m = json.load(open(f"{R}/kp_sd1_meta.json")); TM = [p for p in m["phases"] if p.get("matched_at")][0]["matched_at"]
    tr, tk = (pd.read_csv(f"{R}/{x}_train.tsv", sep="\t").set_index("step").loss for x in ("refL_sd1", "kp_sd1"))
    c = tk.index.intersection(tr.index); d = (tk[c] - tr[c]).sort_index(); t = d.index.values - TM; y = d.values[t >= 1]; t = t[t >= 1]
    ys = np.convolve(np.r_[[y[0]] * 4, y, [y[-1]] * 4], np.ones(9) / 9, "valid"); w = (t >= 10) & (t <= 150)
    i = np.nonzero(w)[0][np.argmax(ys[w])]; print(f"seed 1 training loss (9-step mean): maximum {ys[i]:+.4f} at {t[i]:g} steps")
except FileNotFoundError:
    pass

print("== token bins: per-token maximum of the rarest bin (rank > 10 000) over that of the ten most frequent targets")
fr = {"A": [], "B": []}
for run in ("kp_ta100", "kp_ta1000", "kp_h2", "kp_l2", *[f"kp_sd{s}" for s in range(1, 6)]):
    d = J(f"k_detail_{run}.json")
    if d is None: print(f"{run}: pending"); continue
    row = []
    for S in ("A", "B"):
        if S not in d: continue
        b = d[S]["fbin_stats"]; q = b[-1]["peak"] / b[0]["peak"]; fr[S].append(q)
        am = d[S]["fbin_at_match"]; lag = [x["t_peak"] for x in b]
        row.append(f"{S} {q:.2f} (at match: max |dL| {max(map(abs, am)):.3f}, signs {''.join('+' if x > 0 else '-' for x in am)}; lags {lag})")
    print(f"{run}: " + "; ".join(row))
for S in ("A", "B"):
    if fr[S]: print(f"set {S}: {len(fr[S])} runs, factors {min(fr[S]):.2f}-{max(fr[S]):.2f}, all above 1: {min(fr[S]) > 1}")

print("== breadth, replicates and reverse runs: set A from k_table.json; set B with the same statistics")
TB = {r["kov"]: r for r in (J("k_table.json") or [])}
SB = lambda x: None if x is None else f"{x:+.4f}"
for kov in ("kp", "kp_sd1", "code_kp", "pubmed_kp", "big_kp", "kp_wd0", "km_wd0", "kpA8", "kp_r1", "kp_r2", "km2",
            *[f"km_sd{s}" for s in range(1, 6)], "code_km", "pubmed_km"):
    w = TB.get(kov)
    if w is None or w.get("status") != "ok": print(f"{kov}: {None if w is None else w.get('status')}"); continue
    m = json.load(open(f"{R}/{kov}_meta.json")); TM = [p for p in m["phases"] if p.get("matched_at")][0]["matched_at"]
    k, r = (pd.read_csv(f"{R}/{x}_eval.tsv", sep="\t") for x in (kov, w["ref"])); b = "; B: not recorded"
    if "eval_loss_b" in k and "eval_loss_b" in r:                  # runs that predate set B have no B column
        k, r = (x.dropna(subset=["eval_loss_b"]).sort_values("step") for x in (k, r))
        kk = k[(k.step >= TM) & (k.step <= r.step.max())]; t = (kk.step.values - TM).astype(float)
        if len(t) and t.max() >= 150:
            ys = pd.Series(kk.eval_loss_b.values - np.interp(kk.step, r.step, r.eval_loss_b)).rolling(3, center=True, min_periods=1).mean().values
            ww = (t >= 10) & (t <= 150); i = np.nonzero(ww)[0][np.argmax(ys[ww])]; j = np.nonzero(ww)[0][np.argmax(np.abs(ys[ww]))]
            b = f"; B: peak {ys[i]:+.4f} at {t[i]:g}, extremum {ys[j]:+.4f} at {t[j]:g}"
    if w["protocol"] == "cool-heat":                               # the early dip of the reverse protocol (unsmoothed, set A)
        k, r = (pd.read_csv(f"{R}/{x}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step") for x in (kov, w["ref"]))
        kk = k[(k.step > TM) & (k.step <= TM + 20)]; y = kk.eval_loss.values - np.interp(kk.step, r.step, r.eval_loss)
        if len(y): b += f"; A minimum over 1-20 steps {y.min():+.4f} at {kk.step.values[np.argmin(y)] - TM}"
        if "eval_loss_b" in k and "eval_loss_b" in r:
            kb, rb = (x.dropna(subset=["eval_loss_b"]) for x in (k, r)); kb = kb[(kb.step > TM) & (kb.step <= TM + 20)]
            yb = kb.eval_loss_b.values - np.interp(kb.step, rb.step, rb.eval_loss_b)
            if len(yb): b += f"; B minimum over 1-20 steps {yb.min():+.4f} at {kb.step.values[np.argmin(yb)] - TM}"
    print(f"{kov} ({w['system']}, {w['protocol']}): match after {w['match_steps']} steps, dA {w['dA']:.0f}, L_ref {w['L_ref']:.3f}, "
          f"gap at match A {SB(w['gap_A'])} B {SB(w['gap_B'])}; A: peak {SB(w['peak'])} at {w['t_peak']} "
          f"({100 * w['peak'] / w['L_ref']:.1f}% of L_ref), extremum {SB(w['ext'])} at {w['t_ext']}, area {w['area']}, "
          f"late {SB(w['late'])}, w_rel {w.get('w_rel')}" + b)
ks = [TB[f"km_sd{s}"] for s in range(1, 6) if TB.get(f"km_sd{s}", {}).get("status") == "ok"]
if len(ks) > 1:
    for q in ("ext", "t_ext", "late", "dA"):
        v = np.array([x[q] for x in ks if x[q] is not None], float)
        print(f"km seeds (n={len(v)}) {q}: mean {v.mean():+.4f} s.e.m. {v.std(ddof=1) / np.sqrt(len(v)):.4f} range {v.min():+.4f} to {v.max():+.4f}")

P = J("k_peaklaw_status.json")
if P:
    print(f"== peak-time law: {P['n_scored']} scored of {P['n_planned']} planned; status {P['n_by_status']}")
    for k in ("law", "constant_lag", "constant_fraction"):
        a, b = P[f"{k}_t_star"], P[f"{k}_t_max"]
        print(f"{k}: t* {a['pass']}/{a['n']} (factor {a['factor']:.3f}); t_max {b['pass']}/{b['n']} (factor {b['factor']:.3f})")
    print("unscored:", [r for r, v in P["runs"].items() if v["status"] != "scored"])

C = J("k_contest.json")
if C:
    W = [x for x in C["runs"] if x["kind"] != "exploratory"]
    print(f"== contest: {len({x['run'] for x in W})} scored withheld Kovacs runs; passes", {m: f"{c['pass']}/{c['n']}" for m, c in C["counts"].items()})
    for m in ("nqm2pq", "nqmS"):
        v = [x for x in W if x["model"] == m]; h = [x["height_ratio"] for x in v]; tr = [x["time_ratio"] for x in v]
        ok = lambda q: 1 / 1.5 < q < 1.5
        print(f"{m}: height {min(h):.2f}-{max(h):.2f} (within 1.5: {sum(map(ok, h))}), time {min(tr):.2f}-{max(tr):.2f} "
              f"(within: {sum(map(ok, tr))}), NRMSE<0.3: {sum(x['nrmse'] < 0.3 for x in v)} of {len(v)}")

A = J("k_anneal_rep.json")
if A:
    print("== anneal replications:", {k: v for k, v in A.items() if k != "pairs"})
    ci = lambda v, k, d=3: f"{v[k]:.{d}f} ({v[k + '_ci'][0]:.{d}f}-{v[k + '_ci'][1]:.{d}f})"
    for s, p in A["pairs"].items():
        print(f"seed {s}: {p.get('outcome')} {p.get('status', '')} robust_agree={p.get('robust_agree')}")
        if "A" not in p: continue
        print(f"  match registered {p['match_registered']} replay {p['match_replay']}")
        for S in ("A", "B"):
            v = p[S]
            print(f"  {S}: start {v['start']:+.4f}; final {ci(v, 'G_last')}; last3 {ci(v, 'G_last3')}; over clock {ci(v, 'D_clock')};"
                  f" over free max {ci(v, 'D_max')}; outcome {v['outcome']}/{v['outcome_last3']} clock {v['clock']} max {v['max']}")

Q = J("k_reset_rule.json")
if Q:
    print("== resets:", {k: (v.get("class"), v.get("status")) for k, v in Q.items() if isinstance(v, dict)})
