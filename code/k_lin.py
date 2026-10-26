# E10 linear response (Codex eval #3).
#  proportionality: excursion of each constant-lr step run divided by (eta/eta_ref - 1), against steps and against intrinsic
#                   time (cumulative eta/eta_ref since the jump); collapse in the linear regime
#  superposition:   pulse_two (heat pulse, rest, cold pulse) against pulse_hot + pulse_cold_late (the same two pulses alone)
# Usage: python -I k_lin.py <results_dir> <ref_tag> <step_tags,comma> [out.json]
import sys, json, numpy as np, pandas as pd

R, ref_tag, steps = sys.argv[1], sys.argv[2], [t for t in sys.argv[3].split(",") if t]
out = sys.argv[4] if len(sys.argv) > 4 else f"{R}/k_lin.json"
ER = 1e-3
ld = lambda t: pd.read_csv(f"{R}/{t}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
ref = ld(ref_tag); s0 = int(ref.step.min())


def exc(tag):
    d = ld(tag); d = d[(d.step >= s0) & (d.step <= ref.step.max())]
    return d.step.values - s0, d.eval_loss.values - np.interp(d.step, ref.step, ref.eval_loss)


res = {"steps": {}}
grid = np.array([5, 10, 20, 50, 100, 200, 300, 500, 1000, 2000])
for tag in steps:
    try:
        t, y = exc(tag)
    except FileNotFoundError:
        continue
    lr = json.load(open(f"{R}/{tag}_meta.json"))["phases"][0]["lr"] / ER
    res["steps"][tag] = {"eta": lr, "t": t.tolist(), "dl": y.tolist(), "per_unit": (y / (lr - 1)).tolist(),
                         "at": {int(g): float(np.interp(g, t, y) / (lr - 1)) for g in grid},
                         "at_intrinsic": {int(g): float(np.interp(g / lr, t, y) / (lr - 1)) for g in grid}}
    print(tag, lr, {g: round(v, 4) for g, v in res["steps"][tag]["at"].items()}, flush=True)
try:
    t2, y2 = exc("pulse_two"); th, yh = exc("pulse_hot"); tc, yc = exc("pulse_cold_late")
    tt = t2[t2 <= min(th.max(), tc.max())]; s = np.interp(tt, th, yh) + np.interp(tt, tc, yc); o = np.interp(tt, t2, y2)
    res["superposition"] = {"t": tt.tolist(), "obs": o.tolist(), "sum": s.tolist(),
                            "rms_dev": float(np.sqrt(np.mean((o - s)[tt >= 600] ** 2))), "range": float(np.ptp(o[tt >= 600]))}
    print("superposition rms dev after the cold pulse", res["superposition"]["rms_dev"], "range", res["superposition"]["range"])
except FileNotFoundError:
    pass
json.dump(res, open(out, "w"))
