# E10 held-out detail of a Kovacs run against its reference (both written by kov_train.py with <tag>_detail.npz):
#   * sequence bootstrap (blocks of 8 consecutive validation sequences, 1000 resamples) of the excursion
#     dL(t) = L_run(t) - L_ref(t) on the matched set A and the untouched set B, for statistics fixed in advance:
#     peak = max of the 3-eval running mean over 10..150 steps after release, area = integral over 0..300 steps,
#     late = mean over 1000..2000 steps after release
#   * the excursion split by frequency-rank bin of the target token and by context position (mode-resolved view of the
#     matched state: at the match the total gap is zero, the parts need not be)
#   * per-matrix weight norms at the match
# Usage: python -I k_detail.py <results_dir> <ref_tag> <tag> [out.json]
import sys, json, numpy as np

R, rt, tg = sys.argv[1:4]; out = sys.argv[4] if len(sys.argv) > 4 else f"{R}/k_detail_{tg}.json"
dr, dt = np.load(f"{R}/{rt}_detail.npz"), np.load(f"{R}/{tg}_detail.npz")
ph = [p for p in json.load(open(f"{R}/{tg}_meta.json"))["phases"] if p.get("matched_at")]
TM = ph[0]["matched_at"] if ph else None
rs, ts = dr["step"], dt["step"]
keep = (ts >= rs.min()) & (ts <= rs.max()); ts = ts[keep]
POS = [(0, 8), (8, 64), (64, 256), (256, 512)]
rng = np.random.default_rng(0)


def interp_rows(x_ref, s):
    """reference rows (eval x features) linearly interpolated to steps s"""
    i = np.clip(np.searchsorted(rs, s), 1, len(rs) - 1); w = ((s - rs[i - 1]) / (rs[i] - rs[i - 1]))[:, None]
    return x_ref[i - 1] * (1 - w) + x_ref[i] * w


def stats(t, y):
    ys = np.convolve(np.r_[y[0], y, y[-1]], np.ones(3) / 3, "valid")
    w = (t >= 10) & (t <= 150); a = (t >= 0) & (t <= 300); l = (t >= 1000) & (t <= 2000)
    return {"peak": float(ys[w].max()) if w.any() else None, "t_peak": int(t[w][np.argmax(ys[w])]) if w.any() else None,
            "area": float(np.trapezoid(y[a], t[a])) if a.sum() > 1 else None, "late": float(y[l].mean()) if l.any() else None}


res = {"ref": rt, "tag": tg, "matched_at": TM}
for S in ("A", "B"):
    if f"{S}_seq" not in dt.files or f"{S}_seq" not in dr.files: continue
    d = dt[f"{S}_seq"][keep] - interp_rows(dr[f"{S}_seq"], ts)                 # eval x sequence
    t = ts - (TM if TM else ts[0]); y = d.mean(1)
    r = {"curve": [[int(a), float(b)] for a, b in zip(t, y)]}
    if TM:
        r.update(stats(t, y))
        nb = d.shape[1] // 8; blocks = d[:, :nb * 8].reshape(len(t), nb, 8)
        bs = [stats(t, blocks[:, rng.integers(0, nb, nb)].mean((1, 2))) for _ in range(1000)]
        for k in ("peak", "area", "late"):
            v = np.array([b[k] for b in bs if b[k] is not None])
            if len(v): r[f"{k}_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]; r[f"{k}_se"] = float(v.std())
        r["gap_at_match"] = float(np.interp(0, t, y))
    if f"{S}_fb" in dt.files and f"{S}_fb" in dr.files:
        cnt = dt[f"{S}_fbcount"]; fb = (dt[f"{S}_fb"][keep] - interp_rows(dr[f"{S}_fb"], ts)) / cnt
        r["fbin_edges"] = dt["fb_edges"].tolist(); r["fbin_weight"] = (cnt / cnt.sum()).tolist()
        r["fbin_curves"] = fb.T.tolist()
        if TM:
            r["fbin_at_match"] = [float(np.interp(0, t, fb[:, j])) for j in range(fb.shape[1])]
            r["fbin_stats"] = [stats(t, fb[:, j]) for j in range(fb.shape[1])]
    pz = dt[f"{S}_pos"][keep] - interp_rows(dr[f"{S}_pos"], ts)
    r["pos_groups"] = POS; r["pos_curves"] = [pz[:, a:b].mean(1).tolist() for a, b in POS]
    if TM:
        r["pos_at_match"] = [float(np.interp(0, t, pz[:, a:b].mean(1))) for a, b in POS]
        r["pos_stats"] = [stats(t, pz[:, a:b].mean(1)) for a, b in POS]
    r["t"] = t.tolist(); res[S] = r
    print(S, json.dumps({k: v for k, v in r.items() if not k.endswith("curves") and k not in ("curve", "t")}), flush=True)
if TM:
    ln_t, ln_r = dt["lnorm"][keep], interp_rows(dr["lnorm"], ts)
    i = int(np.argmin(np.abs(ts - TM)))
    res["lnorm_rel_at_match"] = dict(zip(dt["lnorm_names"].tolist(), (ln_t[i] / ln_r[i] - 1).tolist()))
json.dump(res, open(out, "w"))
