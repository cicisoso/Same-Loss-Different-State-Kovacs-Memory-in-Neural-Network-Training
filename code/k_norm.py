# E10 norm interventions at the match (PREREG addendum 2026-10-20 06:51). Every branch is compared with the unchanged reference
# continuation nm_ref_sham on identical data; Kovacs branches are also compared with the unchanged Kovacs continuation
# nm_kp_sham. Reports the registered statistics (peak = max of the 3-evaluation running mean over 10-150 steps, area over
# 0-300 steps, late = mean over 1000-1500 steps), the registered verdicts, function preservation (held-out loss and
# per-sequence losses at the first evaluation, before any update, against the sham from the same checkpoint), angular
# updates per matrix group (mean over 1-100 and 1000-1500 steps) and, where the transplant runs exist, the run-to-run
# reproducibility of identical continuations (nm_*_sham against sw_*_own).
# Usage: python -I k_norm.py <results_dir> [out.json]
import os, sys, json, numpy as np, pandas as pd

R = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else f"{R}/k_norm.json"
KB = ["nm_kp_sham", "nm_kp_proj", "nm_kp_resid", "nm_kp_both"]; RB = ["nm_ref_proj", "nm_ref_resid"]


def ld(tag, f="eval"):
    try:
        return pd.read_csv(f"{R}/{tag}_{f}.tsv", sep="\t").sort_values("step")
    except (FileNotFoundError, pd.errors.EmptyDataError):    # not started, or a running branch has not written a row yet
        return None


def stats(t, y):
    ys = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values
    w = (t >= 10) & (t <= 150); a = (t >= 0) & (t <= 300); l = (t >= 1000) & (t <= 1500)
    out = {"peak": None, "t_peak": None, "area": None, "late": None}    # a window not yet covered is missing, not partial
    if t.max() >= 150:
        i = np.nonzero(w)[0][np.argmax(ys[w])]; out.update(peak=float(ys[i]), t_peak=float(t[i]))
    if t.max() >= 300: out["area"] = float(np.trapezoid(y[a], t[a]))
    if t.max() >= 1500: out["late"] = float(y[l].mean())
    return out


base = ld("nm_ref_sham"); res = {"branches": {}}
if base is None: raise SystemExit("nm_ref_sham missing")
base = base.dropna(subset=["eval_loss"]); t0 = int(base.step.min()); res["t0"] = t0
for tag in KB + RB:
    d = ld(tag)
    if d is None: continue
    d = d.dropna(subset=["eval_loss"]); d = d[d.step <= base.step.max()]
    t = (d.step.values - t0).astype(float); rec = {"t": t.tolist()}
    for col, key in (("eval_loss", "A"), ("eval_loss_b", "B")):
        if col in d and d[col].notna().all():
            y = d[col].values - np.interp(d.step.values, base.step.values, base[col].values)
            rec[key] = y.tolist(); rec[f"stats_{key}"] = stats(t, y)
    rec["wnorm_rel"] = (d.wnorm.values / np.interp(d.step.values, base.step.values, base.wnorm.values) - 1).tolist()
    sham = "nm_kp_sham" if tag.startswith("nm_kp") else "nm_ref_sham"     # function preservation at the first evaluation
    s0 = ld(sham)
    if s0 is not None and tag != sham:
        rec["first_eval_diff"] = float(d.eval_loss.values[0] - s0.dropna(subset=["eval_loss"]).eval_loss.values[0])
        try:
            a_, b_ = np.load(f"{R}/{tag}_detail.npz"), np.load(f"{R}/{sham}_detail.npz")
            dd = a_["A_seq"][0] - b_["A_seq"][0]
            rec["first_eval_seq_absdiff_mean"] = float(np.abs(dd).mean()); rec["first_eval_seq_absdiff_max"] = float(np.abs(dd).max())
        except (FileNotFoundError, KeyError):
            pass
    an = ld(tag, "angle")
    if an is not None:
        ta = an.step.values - t0
        rec["angle"] = {w: {c[4:]: float(an[c][(ta >= lo) & (ta <= hi)].mean()) for c in an.columns if c.startswith("ang_")}
                        for w, (lo, hi) in (("early", (1, 100)), ("late", (1000, 1500)))}
    res["branches"][tag] = rec
    print(tag, "A", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.get("stats_A", {}).items()}),
          "first-eval diff", round(rec.get("first_eval_diff", 0.0), 5), flush=True)

# registered verdicts
B = res["branches"]; S = lambda tag, k: B.get(tag, {}).get("stats_A", {}).get(k)
ver = {}
cmp = lambda a, b, f: None if a is None or b is None else f(a, b)          # None while either window is incomplete
for tag in ("nm_kp_proj", "nm_kp_resid", "nm_kp_both"):
    if tag not in B: continue
    ver[tag] = {"area_up": cmp(S(tag, "area"), S("nm_kp_sham", "area"), lambda a, b: a > b),
                "peak_up": cmp(S(tag, "peak"), S("nm_kp_sham", "peak"), lambda a, b: a > b),
                "late_less_negative": cmp(S(tag, "late"), S("nm_kp_sham", "late"), lambda a, b: a > b),
                "d_area": cmp(S(tag, "area"), S("nm_kp_sham", "area"), lambda a, b: a - b),
                "d_peak": cmp(S(tag, "peak"), S("nm_kp_sham", "peak"), lambda a, b: a - b),
                "d_late": cmp(S(tag, "late"), S("nm_kp_sham", "late"), lambda a, b: a - b)}
for tag in ("nm_ref_proj", "nm_ref_resid"):
    if tag not in B: continue
    ver[tag] = {"area_negative": None if S(tag, "area") is None else S(tag, "area") < 0,
                "late_negative": None if S(tag, "late") is None else S(tag, "late") < 0,
                "area": S(tag, "area"), "late": S(tag, "late")}
for tag in ver: ver[tag]["complete"] = os.path.exists(f"{R}/{tag}_meta.json")
res["verdicts"] = ver
print("verdicts", json.dumps(ver, default=float), flush=True)

# reproducibility of identical continuations
for a_, b_ in (("nm_kp_sham", "sw_kp_own"), ("nm_ref_sham", "sw_ref_own")):
    x, y = ld(a_), ld(b_)
    if x is None or y is None: continue
    x, y = x.dropna(subset=["eval_loss"]), y.dropna(subset=["eval_loss"]); st = np.intersect1d(x.step, y.step)
    dv = x.set_index("step").loc[st].eval_loss.values - y.set_index("step").loc[st].eval_loss.values
    res[f"repro_{a_}"] = {"rms": float(np.sqrt(np.mean(dv ** 2))), "max": float(np.abs(dv).max()), "n": int(len(st))}
    print("reproducibility", a_, "vs", b_, res[f"repro_{a_}"], flush=True)
json.dump(res, open(out, "w"))
