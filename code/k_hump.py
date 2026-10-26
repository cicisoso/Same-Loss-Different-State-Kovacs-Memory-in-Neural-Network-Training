# E10: shape of the Kovacs excursion after release. For each run, dL(t) = loss - reference (interpolated) for t >= release,
# fitted with the two-mode form  dL = A (exp(-t/ts) - exp(-t/tf)) + C ; the predicted peak time is
# t* = tf ts / (ts - tf) ln(ts / tf). Peak height and time are also read from the data smoothed over 3 evaluations.
# Uncertainty: moving-block bootstrap of the residuals (blocks of 5 evaluations, 200 resamples).
# Usage: python -I k_hump.py <results_dir> <ref_tag> <tag[,tag...]> [out.json]
import sys, json, numpy as np, pandas as pd
from scipy.optimize import curve_fit

R, ref_tag, tags = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
out = sys.argv[4] if len(sys.argv) > 4 else f"{R}/k_hump.json"
ref = pd.read_csv(f"{R}/{ref_tag}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
rng = np.random.default_rng(5)
f2 = lambda t, A, ltf, lts, C: A * (np.exp(-t / np.exp(lts)) - np.exp(-t / np.exp(ltf))) + C
res = {}
for tag in tags:
    try:
        d = pd.read_csv(f"{R}/{tag}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]); m = json.load(open(f"{R}/{tag}_meta.json"))
    except FileNotFoundError:
        continue
    ph = [p for p in m["phases"] if p.get("matched_at")]
    if not ph: res[tag] = {"matched": False}; print(tag, "no match", flush=True); continue
    t0 = ph[0]["matched_at"]
    d = d[(d.step >= t0) & (d.step <= ref.step.max())]
    t = (d.step - t0).values.astype(float); y = d.eval_loss.values - np.interp(d.step, ref.step, ref.eval_loss)
    sgn = 1.0 if y[np.argmax(np.abs(y[: max(3, len(y) // 4)]))] > 0 else -1.0
    ys = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values
    ip = int(np.argmax(sgn * ys))
    r = {"matched_at": int(t0), "gap": float(y[0]), "sign": sgn, "peak": float(ys[ip]), "t_peak": float(t[ip]), "n": int(len(t))}
    try:
        p0 = [sgn * abs(ys[ip]) * 1.5, np.log(10), np.log(300), 0.0]
        popt, _ = curve_fit(f2, t, y, p0=p0, maxfev=50000)
        A, tf, ts, C = popt[0], np.exp(popt[1]), np.exp(popt[2]), popt[3]
        if tf > ts: tf, ts, A = ts, tf, -A
        tstar = tf * ts / (ts - tf) * np.log(ts / tf)
        resid = y - f2(t, *popt); fit = f2(t, *popt); boots = []
        for _ in range(200):
            idx = np.concatenate([np.arange(s, min(s + 5, len(t))) for s in rng.integers(0, len(t), len(t) // 5 + 1)])[: len(t)]
            try:
                pb, _ = curve_fit(f2, t, fit + resid[idx], p0=popt, maxfev=20000)
                tfb, tsb = sorted([np.exp(pb[1]), np.exp(pb[2])])
                boots.append([f2(t, *pb)[np.argmax(sgn * f2(t, *pb))] - pb[3], tfb * tsb / (tsb - tfb) * np.log(tsb / tfb)])
            except Exception:
                pass
        boots = np.array(boots)
        hump = float(fit[np.argmax(sgn * fit)] - C)
        r.update({"A": float(A), "tau_f": float(tf), "tau_s": float(ts), "C": float(C), "t_star": float(tstar),
                  "hump_fit": hump, "hump_se": float(boots[:, 0].std()) if len(boots) > 10 else None,
                  "t_star_se": float(boots[:, 1].std()) if len(boots) > 10 else None,
                  "rmse": float(np.sqrt(np.mean(resid ** 2)))})
    except Exception as e:
        r["fit_error"] = str(e)
    res[tag] = r
    print(tag, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(res, open(out, "w"), indent=1)
