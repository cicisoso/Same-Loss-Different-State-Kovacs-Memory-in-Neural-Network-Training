# E10 anneal endpoints against the free excursion (PREREG addendum 2026-10-20 10:20). For a matched pair annealed with the
# 500-step linear decay (kov and ref anneal runs) and the free continuation of the same or a replayed pair at eta_ref (free kov
# and ref runs, from their match), every statistic is computed from the per-sequence held-out losses of the detail logs and
# bootstrapped jointly: the same validation blocks (8 sequences, 1000 resamples) are drawn for all four runs, so that
#   final   = gap at the last anneal evaluation (and the mean of the last three),
#   fmax    = largest free excursion over the evaluations 0-1500 steps after the free match (recomputed in each resample),
#   clock   = free excursion at intrinsic time B + 250.5 (the shared-clock prediction; 250.5 reference steps of the discrete
#             schedule), with the free reference interpolated per sequence to the free Kovacs run's evaluation steps,
# and final - fmax and final - clock carry the covariance of the shared sequences.
# Usage: python -I k_anneal_pairs.py <results_dir> <name=an_kov:an_ref:free_kov:free_ref:B,...> <out.json>
#   free runs start at their match (matched_at in the free Kovacs run's meta, else the first common step).
# A pair is scored ("status": "ok") only if all four runs have finished (meta), the anneal covers B + 500 steps, the free pair
# covers the shared-clock time and the 1500 steps of the free maximum, and the losses are finite with the same sequences
# (same number per set and, where all four detail logs store them, the same target-token counts per frequency bin, a
# fingerprint of the evaluated sequences; the detail logs carry no sequence identifiers).
import sys, os, json, numpy as np

R = sys.argv[1]; specs = [x.split("=") for x in sys.argv[2].split(",") if x]; out = sys.argv[3]
NB, BL = 1000, 8


def det(tag):
    d = np.load(f"{R}/{tag}_detail.npz")
    return {k: d[k] for k in ("step", "A_seq", "B_seq", "A_fbcount", "B_fbcount") if k in d.files}


def matched_at(tag):
    try:
        m = json.load(open(f"{R}/{tag}_meta.json"))
        return [p for p in m["phases"] if p.get("matched_at")][0]["matched_at"]
    except (FileNotFoundError, IndexError, KeyError):
        return None


def interp_rows(st_src, X, st_dst):
    """per-sequence linear interpolation of X (n_eval x n_seq) from steps st_src to steps st_dst"""
    st_src = np.asarray(st_src, float); j = np.clip(np.searchsorted(st_src, st_dst) - 1, 0, len(st_src) - 2)
    w = (np.asarray(st_dst, float) - st_src[j]) / (st_src[j + 1] - st_src[j]); w = np.clip(w, 0, 1)[:, None]
    return X[j] * (1 - w) + X[j + 1] * w


rng = np.random.default_rng(0); res = {}
for name, spec in specs:
    ak, ar, fk, fr, B = spec.split(":"); B = int(B)
    try:
        AK, AR, FK, FR = det(ak), det(ar), det(fk), det(fr)
    except FileNotFoundError as e:
        res[name] = {"status": f"missing {e.filename}"}; print(name, "missing", e.filename); continue
    if not (len(AK["step"]) and AK["step"][-1] == AR["step"][-1]):
        res[name] = {"status": "anneal runs incomplete or misaligned"}; print(name, "incomplete"); continue
    common = sorted(set(AK["step"].tolist()) & set(AR["step"].tolist()))
    ik = [list(AK["step"]).index(s) for s in common]; ir = [list(AR["step"]).index(s) for s in common]
    t0f = matched_at(fk) or int(max(FK["step"].min(), FR["step"].min()))
    mk = (FK["step"] >= t0f) & (FK["step"] <= min(FR["step"].max(), t0f + 1500)); sf = FK["step"][mk]; tf = sf - t0f
    bad = [f"{t} has no meta (not finished)" for t in (ak, ar, fk, fr) if not os.path.exists(f"{R}/{t}_meta.json")]
    if common[-1] - common[0] != B + 500:                       # B steps at eta_ref, then the scheduled 500-step decay
        bad.append(f"anneal covers {common[-1] - common[0]} steps, expected {B + 500}")
    if not len(tf) or tf.max() < B + 250.5:
        bad.append("free pair does not cover the shared-clock time (no extrapolation)")
    elif tf.max() < 1500:
        bad.append(f"free pair covers {int(tf.max())} of the 1500 steps of the free maximum")
    for S in ("A", "B"):
        shp = {X[f"{S}_seq"].shape[1] for X in (AK, AR, FK, FR)}
        if len(shp) != 1: bad.append(f"set {S}: different numbers of sequences {sorted(shp)}")
        elif not all(np.isfinite(X[f"{S}_seq"]).all() for X in (AK, AR, FR)) or not np.isfinite(FK[f"{S}_seq"][mk]).all():
            bad.append(f"set {S}: non-finite losses")
        fb = [X.get(f"{S}_fbcount") for X in (AK, AR, FK, FR)]
        if all(c is not None for c in fb) and not all(np.array_equal(c, fb[0]) for c in fb):
            bad.append(f"set {S}: different target-token counts (different sequences)")
    if bad:
        res[name] = {"status": "invalid: " + "; ".join(bad)}; print(name, res[name]["status"]); continue
    e = {"pair": [ak, ar], "free": [fk, fr], "B": B, "free_match": int(t0f), "anneal_start": int(common[0]),
         "anneal_steps": int(common[-1] - common[0])}
    for S in ("A", "B"):
        n = AK[f"{S}_seq"].shape[1]; nb = n // BL
        dA = AK[f"{S}_seq"][ik] - AR[f"{S}_seq"][ir]                       # anneal gap per evaluation and sequence
        dF = FK[f"{S}_seq"][mk] - interp_rows(FR["step"], FR[f"{S}_seq"], sf)   # free excursion per evaluation and sequence
        blk = lambda X: X[:, :nb * BL].reshape(X.shape[0], nb, BL).mean(2)        # evaluation x block means
        bA, bF = blk(dA), blk(dF)
        last, last3 = bA[-1], bA[-3:].mean(0)
        u = B + 250.5; jj = np.clip(np.searchsorted(tf, u) - 1, 0, len(tf) - 2); ww = (u - tf[jj]) / (tf[jj + 1] - tf[jj])
        clk = bF[jj] * (1 - ww) + bF[jj + 1] * ww
        stat = lambda c: {"final_last": c @ last, "final_last3": c @ last3, "fmax": (bF @ c).max(), "clock": c @ clk}
        point = stat(np.full(nb, 1 / nb))
        boots = [stat(rng.multinomial(nb, np.full(nb, 1 / nb)) / nb) for _ in range(NB)]
        ci = lambda f: [float(np.percentile([f(b) for b in boots], 2.5)), float(np.percentile([f(b) for b in boots], 97.5))]
        for k_ in ("final_last", "final_last3", "fmax", "clock"):
            e[f"{S}_{k_}"] = float(point[k_]); e[f"{S}_{k_}_ci"] = ci(lambda b, k_=k_: b[k_])
        for k_ in ("final_last", "final_last3"):
            e[f"{S}_{k_}_minus_fmax"] = float(point[k_] - point["fmax"])
            e[f"{S}_{k_}_minus_fmax_ci"] = ci(lambda b, k_=k_: b[k_] - b["fmax"])
            e[f"{S}_{k_}_minus_clock"] = float(point[k_] - point["clock"])
            e[f"{S}_{k_}_minus_clock_ci"] = ci(lambda b, k_=k_: b[k_] - b["clock"])
        e[f"{S}_fmax_step"] = int(tf[np.argmax(bF.mean(1))])
        e[f"{S}_pre_anneal"] = float(np.interp(common[0] + B, common, dA.mean(1)))   # gap when the decay starts
    e["status"] = "ok"; res[name] = e
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v] if isinstance(v, list) and
                                k.endswith("ci") else v) for k, v in e.items() if k.startswith("A_") or k in ("free_match", "B")}),
          flush=True)
json.dump(res, open(out, "w"))
