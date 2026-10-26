# E10 linear-regime runs (PREREG: steps and 300-step pulses at 1.2 and 0.8 eta_ref; two-pulse superposition), summarized
# for the Supporting Information from the excursions against the reference (k_lin.py definitions, no fitting):
#  steps:  per-unit response dL / (eta/eta_ref - 1) of the +20% and -20% steps at intrinsic times s = t * eta/eta_ref, and
#          their largest relative difference over 10 <= s <= 1000; the same for the eta_ref/2 and 2 eta_ref steps
#  pulses: per-unit response after the end of the +20% and -20% pulses (t = 300 + u), and the largest difference
#  superposition: pulse_two against pulse_hot + pulse_cold_late after the start of the cold pulse (t >= 600): rms deviation,
#          range of the observed excursion, and the deviation at the largest observed |excursion|
#  steps and pulses are also compared at common nominal times; every interpolated time lies inside the evaluated range
#  frozen models (optional): whole-run absolute-loss RMSE of the registered schedule-to-loss fits (k_models.py on the
#          registered fit set) on these withheld runs, and whether their parameters equal the registered fits
# Usage: python -I k_lin_report.py <results_dir> [ref_tag=ref] [k_models_lin.json] [k_models_withheld.json]
import sys, json, numpy as np, pandas as pd

R = sys.argv[1]; REF = sys.argv[2] if len(sys.argv) > 2 else "ref"
MOD = sys.argv[3] if len(sys.argv) > 3 else None; REG = sys.argv[4] if len(sys.argv) > 4 else None
ld = lambda t: pd.read_csv(f"{R}/{t}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
ref = ld(REF); s0 = int(ref.step.min())


def exc(tag):
    d = ld(tag); d = d[(d.step >= s0) & (d.step <= ref.step.max())]
    return (d.step.values - s0).astype(float), d.eval_loss.values - np.interp(d.step, ref.step, ref.eval_loss)


def lr0(tag): return json.load(open(f"{R}/{tag}_meta.json"))["phases"][0]["lr"] / 1e-3


def at(x, t, y):
    """interpolated excursion, refusing to extrapolate past the evaluated range"""
    x = np.asarray(x, float); assert x.min() >= t.min() and x.max() <= t.max(), (x.min(), x.max(), t.min(), t.max())
    return np.interp(x, t, y)


def reldiff(a, b): return np.abs(a - b) / ((np.abs(a) + np.abs(b)) / 2)


S = np.array([10, 30, 100, 300, 1000])
print("== steps: per-unit response at intrinsic time s (nats per unit level)")
pu = {}
for tag in ("step_8e-4", "step_1.2e-3", "step_5e-4", "step_2e-3"):
    try:
        t, y = exc(tag)
    except FileNotFoundError:
        print(tag, "missing"); continue
    g = lr0(tag); ss = np.geomspace(10, 1000, 41); v = at(ss / g, t, y) / (g - 1)
    pu[tag] = (ss, v, at(ss, t, y) / (g - 1))                                      # intrinsic, nominal
    print(f"{tag} (g={g:g}): intrinsic", {int(s): round(float(at(s / g, t, y) / (g - 1)), 5) for s in S},
          "nominal", {int(s): round(float(at(s, t, y) / (g - 1)), 5) for s in S})
for lo_, hi_, nm in (("step_8e-4", "step_1.2e-3", "+-20% steps"), ("step_5e-4", "step_2e-3", "x0.5 / x2 steps")):
    if lo_ in pu and hi_ in pu:
        for k, clock in ((1, "intrinsic s"), (2, "nominal t")):
            rel = reldiff(pu[lo_][k], pu[hi_][k])
            print(f"{nm}, {clock} = 10-1000: relative difference median {np.median(rel):.4f}, largest {rel.max():.4f} "
                  f"(at {pu[lo_][0][np.argmax(rel)]:.0f})")

print("== pulses (300 steps): per-unit response u steps after the end of the pulse")
pp = {}
for tag in ("pulse_8e-4", "pulse_1.2e-3"):
    try:
        t, y = exc(tag)
    except FileNotFoundError:
        print(tag, "missing"); continue
    g = lr0(tag); uu = np.geomspace(5, 2000, 41); v = at(300 + uu, t, y) / (g - 1); pp[tag] = (uu, v)
    print(f"{tag} (g={g:g}): evaluated at u = {(t[(t >= 300) & (t <= 330)] - 300).astype(int).tolist()} after the pulse; per unit at u =",
          {int(u): round(float(at(300 + u, t, y) / (g - 1)), 4) for u in (0, 5, 10, 20, 30, 100, 300, 1000, 2000)})
if len(pp) == 2:
    a, b = pp["pulse_8e-4"][1], pp["pulse_1.2e-3"][1]; d = np.abs(a - b)
    print(f"+-20% pulses: largest absolute per-unit difference over u = 5-2000: {d.max():.4f} (at u = {pp['pulse_8e-4'][0][np.argmax(d)]:.0f}); "
          f"over u = 10-2000: {d[pp['pulse_8e-4'][0] >= 10].max():.4f}")

print("== superposition: pulse_two against pulse_hot + pulse_cold_late")
try:
    t2, y2 = exc("pulse_two"); th, yh = exc("pulse_hot"); tc, yc = exc("pulse_cold_late")
    tt = t2[(t2 <= min(th.max(), tc.max())) & (t2 >= 600)]
    if len(tt) and tt.max() >= 2000:
        o = at(tt, t2, y2); s = at(tt, th, yh) + at(tt, tc, yc); i = np.argmax(np.abs(o))
        print(f"t = 600-{tt.max():.0f}: rms deviation {np.sqrt(np.mean((o - s) ** 2)):.4f} nats; observed range {np.ptp(o):.4f}; "
              f"largest |observed| {o[i]:+.4f} at t = {tt[i]:.0f} (sum {s[i]:+.4f}); mean deviation {np.mean(o - s):+.4f}")
        for lo_, hi_ in ((600, 900), (900, 1200), (1200, 2000), (2000, 3000)):
            w = (tt >= lo_) & (tt < hi_)
            if w.any(): print(f"  {lo_}-{hi_}: rms {np.sqrt(np.mean((o - s)[w] ** 2)):.5f}, mean obs {o[w].mean():+.4f}, mean sum {s[w].mean():+.4f}, "
                              f"share of the squared deviation {np.sum((o - s)[w] ** 2) / np.sum((o - s) ** 2):.3f} ({w.sum()} evaluations)")
    else:
        print("pulse_two incomplete")
except FileNotFoundError as e:
    print("missing", e)

if MOD:
    print("== frozen schedule-to-loss models on the withheld linear-regime runs (whole-run absolute-loss RMSE, nats)")
    m = json.load(open(MOD)); names = [k[:-len("_fit_rmse")] for k in m if k.endswith("_fit_rmse")]
    if REG:
        g = json.load(open(REG))
        same = all(m[k] == g.get(k) for k in m if k.endswith(("_params", "_fit_rmse", "_pq", "_spectrum")))
        print("parameters identical to the registered fits:", same)
    print("fit-set rmse:", {n: round(m[f"{n}_fit_rmse"], 4) for n in names}, "pulse_hot fit rmse:",
          {n: round(m[f"{n}_fit_by_tag"]["pulse_hot"], 4) for n in names}, "reference-continuation fit rmse:",
          {n: round(m[f"{n}_fit_by_tag"]["ref"], 4) for n in names})
    for tag in m["test_tags"]:
        r = {n: m[tag][n]["rmse_abs"] for n in names}
        print(f"{tag}: {min(r.values()):.4f}-{max(r.values()):.4f} ({min(r, key=r.get)}-{max(r, key=r.get)})",
              {n: round(v, 4) for n, v in r.items()})
