# E10 contest scoring with the pre-registered definitions (PREREG.md): for every test run and model, from the curves stored by
# k_models.py (steps after release, observed excursion, predicted excursion):
#   peak = maximum of the 3-evaluation running mean over 10 <= t <= 150 (same window for observation and prediction),
#   t_peak = its lag, area = trapezoidal integral over 0 <= t <= 300, sign = sign of the observed peak,
#   NRMSE = RMSE(pred - obs) after release / range of the smoothed observed excursion.
# Success (pre-registered): correct sign, peak height and time within a factor 1.5, NRMSE < 0.3, all from the registered
# peak (the maximum). Reported separately as a secondary analysis (dated note in PREREG.md, 2026-10-20 06:55): the same
# criteria on the signed extremum (the larger of maximum and minimum in magnitude), which differs only for excursions
# that are mainly negative.
# Usage: python -I k_score.py <k_models_out.json> [out.json]
import sys, json, numpy as np

src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else src.replace(".json", "_score.json")
D = json.load(open(src))
MODELS = ["tissue", "mpl", "fsl", "nqm2", "nqm2pq", "nqmS"]


def smooth3(y):
    y = np.asarray(y, float); p = np.r_[y[0], y, y[-1]]
    return (p[:-2] + p[1:-1] + p[2:]) / 3


def stats(t, y):
    ys = smooth3(y); w = (t >= 10) & (t <= 150)
    if not w.any(): return None
    i = np.nonzero(w)[0][np.argmax(ys[w])]                          # registered: the maximum
    sg = 1.0 if ys[w][np.argmax(np.abs(ys[w]))] > 0 else -1.0       # secondary: sign of the dominant deviation
    j = np.nonzero(w)[0][np.argmax(sg * ys[w])]
    a = (t >= 0) & (t <= 300)
    return {"peak": float(ys[i]), "t_peak": float(t[i]), "area": float(np.trapezoid(y[a], t[a])),
            "ext": float(ys[j]), "t_ext": float(t[j]), "sign": sg}


def judge(so, sp, nrmse, k, tk):
    r_h = sp[k] / so[k] if so[k] != 0 else np.nan
    r_t = sp[tk] / so[tk] if so[tk] > 0 else np.nan
    ok = bool(np.sign(sp[k]) == np.sign(so[k]) and 1 / 1.5 < r_h < 1.5 and 1 / 1.5 < r_t < 1.5 and nrmse < 0.3)
    return float(r_h), float(r_t), ok


res = {}
for run, rec in D.items():
    if not isinstance(rec, dict) or "matched_at" not in rec: continue
    row = {}
    for m in MODELS:
        if m not in rec or "curve" not in rec[m]: continue
        c = np.array(rec[m]["curve"], float); t, o, p = c[:, 0], c[:, 1], c[:, 2]
        so, sp = stats(t, o), stats(t, p)
        if so is None: continue
        nrmse = float(np.sqrt(np.mean((p - o) ** 2)) / max(np.ptp(smooth3(o)), 1e-12))
        r_h, r_t, ok = judge(so, sp, nrmse, "peak", "t_peak")
        e_h, e_t, ok_e = judge(so, sp, nrmse, "ext", "t_ext")
        row[m] = {"obs": so, "pred": sp, "height_ratio": r_h, "time_ratio": r_t, "nrmse": nrmse, "success": ok,
                  "ext_height_ratio": e_h, "ext_time_ratio": e_t, "success_ext": ok_e}
    res[run] = row
    print(run, " ".join(f"{m}:{v['height_ratio']:.2f}/{v['time_ratio']:.2f}/{v['nrmse']:.2f}{'*' if v['success'] else ''}"
                        for m, v in row.items()), flush=True)
summ = {m: {"n": sum(m in r for r in res.values()), "success": sum(r[m]["success"] for r in res.values() if m in r),
            "success_ext": sum(r[m]["success_ext"] for r in res.values() if m in r)} for m in MODELS}
res["_summary"] = summ; print("summary (n, successes):", summ)
json.dump(res, open(out, "w"), indent=1)
