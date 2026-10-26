# Data for Figure 5 (breadth and sign): Kovacs excursions dL(t) = L_kov - L_ref after release for every system that has a
# reference (heat->cool kp and cool->heat km protocols), with the registered statistics; the CIFAR batch-norm thermostat test
# (continuations of the matched CIFAR pair with and without exchanged filter norms); relative weight norms at release.
# Usage: python fig5_data.py <results_dir> <out.json>
import sys, json, numpy as np, pandas as pd

R, out = sys.argv[1], sys.argv[2]
# system key, label, reference tag, Kovacs tags (kp, km), loss unit
SYSTEMS = [("web", "web text, 30M", ("ref", "refL"), "kp", "km2"), ("web124", "web text, 124M", "big_refL", "big_kp", None),
           ("code", "Python code", "code_refL", "code_kp", "code_km"), ("pubmed", "PubMed", "pubmed_refL", "pubmed_kp", "pubmed_km"),
           ("wd0", "web text, no weight decay", "refL_wd0", "kp_wd0", "km_wd0"),
           ("mfn", "MovieLens, momentum SGD", "mfn_refL", "mfn_kp", "mfn_km"), ("mf0n", "MovieLens, plain SGD", "mf0n_refL", "mf0n_kp", "mf0n_km"),
           ("cifar", "CIFAR-10 CNN, momentum SGD", "cifar_refL", "cifar_kp", "cifar_km"),
           ("cifar0", "CIFAR-10 CNN, plain SGD", "cifar0_refL", "cifar0_kp", "cifar0_km")]


def ld(tag):
    return pd.read_csv(f"{R}/{tag}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")


def stats(t, y):
    ys = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values
    w = (t >= 10) & (t <= 150); a = (t >= 0) & (t <= 300); late = (t >= 1000) & (t <= 2000)
    if not late.any(): late = t >= t.max() - 500
    sg = 1.0 if ys[w][np.argmax(np.abs(ys[w]))] > 0 else -1.0
    i = np.nonzero(w)[0][np.argmax(ys[w])]; j = np.nonzero(w)[0][np.argmax(sg * ys[w])]
    return {"peak": float(ys[i]), "t_peak": float(t[i]), "ext": float(ys[j]), "t_ext": float(t[j]),
            "area": float(np.trapezoid(y[a], t[a])), "late": float(y[late].mean())}


def excursion(ref, kov):
    m = json.load(open(f"{R}/{kov}_meta.json")); k = ld(kov); r = ld(ref)
    TM = [p for p in m["phases"] if p.get("matched_at")][0]["matched_at"]
    k = k[(k.step >= TM) & (k.step <= r.step.max())]
    t = (k.step.values - TM).astype(float); y = k.eval_loss.values - np.interp(k.step, r.step, r.eval_loss)
    e = {"t": t.tolist(), "dA": y.tolist(), "stats": stats(t, y), "TM": int(TM),
         "ref_loss_at_match": float(np.interp(TM, r.step, r.eval_loss)),
         "first_steps": int(m["phases"][0]["end"] - m["phases"][0]["start"]), "second_steps": int(TM - m["phases"][1]["start"])}
    if "wnorm" in k and "wnorm" in r:
        e["wnorm_rel_match"] = float(k.wnorm.values[0] / np.interp(TM, r.step, r.wnorm) - 1)
    if "eval_loss_b" in k and "eval_loss_b" in r and r.eval_loss_b.notna().all() and k.eval_loss_b.notna().all():
        e["dB"] = (k.eval_loss_b.values - np.interp(k.step, r.step, r.eval_loss_b)).tolist()
    return e


res = {"systems": {}}
for key, name, ref, kp, km in SYSTEMS:
    s = {"name": name}
    for i, (prot, tag) in enumerate((("kp", kp), ("km", km))):
        if tag is None: continue
        rt = ref[i] if isinstance(ref, tuple) else ref        # the 30M pilot was matched against ref; later runs against refL
        try:
            s[prot] = excursion(rt, tag)
        except (FileNotFoundError, IndexError):
            continue
        print(key, prot, json.dumps({q: round(v, 4) for q, v in s[prot]["stats"].items()}), "w", round(s[prot].get("wnorm_rel_match", np.nan), 4))
    if "kp" in s or "km" in s: res["systems"][key] = s

# CIFAR thermostat: continuations from the matched pair on identical minibatches, against the reference's own continuation
cn = {}
try:
    b = ld("cifar_cn_ref"); t0 = int(b.step.min())
    for tag in ("cifar_cn_own", "cifar_cn_norm", "cifar_cn_refnorm"):
        try:
            d = ld(tag)
        except FileNotFoundError:
            continue
        d = d[d.step <= b.step.max()]; t = (d.step.values - t0).astype(float)
        y = d.eval_loss.values - np.interp(d.step, b.step, b.eval_loss)
        cn[tag] = {"t": t.tolist(), "dA": y.tolist(), "stats": stats(t, y)}
        if "wnorm" in d: cn[tag]["wnorm_rel"] = (d.wnorm.values / np.interp(d.step, b.step, b.wnorm) - 1).tolist()
        print(tag, json.dumps({q: round(v, 4) for q, v in cn[tag]["stats"].items()}))
except FileNotFoundError:
    pass
res["cifar_thermostat"] = cn
# 2x2 factorial over (non-norm state, filter norms), each cell against the reference's own continuation (RR = 0):
# KK = Kovacs own, KR = Kovacs at reference norms, RK = reference at Kovacs norms
if all(t in cn for t in ("cifar_cn_own", "cifar_cn_norm", "cifar_cn_refnorm")):
    fac = {}
    for st in ("area", "late"):
        KK, KR, RK = (cn[t]["stats"][st] for t in ("cifar_cn_own", "cifar_cn_norm", "cifar_cn_refnorm"))
        fac[st] = {"KK": KK, "KR": KR, "RK": RK, "norm": 0.5 * ((KK - KR) + RK), "state": 0.5 * ((KK - RK) + KR),
                   "interaction": KK - KR - RK,                    # same conventions as k_swap.py
                   # simple effects: the norm exchanged in each background (they differ by the interaction)
                   "norm_K": KK - KR, "norm_R": RK, "state_R": KR, "state_K": KK - RK}
        print("factorial", st, {q: round(v, 4) for q, v in fac[st].items()})
    res["cifar_factorial"] = fac
json.dump(res, open(out, "w"))
