# E10 consequence: two checkpoints with the same held-out loss (the Kovacs run at its match, kpS_end1, and the reference at the
# same step, refS) are annealed (linear decay to 0 over 500 steps) after B further steps at eta_ref. Reports the final loss
# difference on sets A and B, as the last evaluation and as the mean of the last 3 evaluations, each with its own block
# bootstrap over validation sequences (blocks of 8, 1000 resamples) from the per-sequence losses of the same evaluations,
# and the difference just before the anneal starts.
# Usage: python -I k_anneal.py <results_dir> <B,B,...> [out.json]
import sys, json, numpy as np, pandas as pd

R = sys.argv[1]; Bs = [int(b) for b in sys.argv[2].split(",")]; out = sys.argv[3] if len(sys.argv) > 3 else f"{R}/k_anneal.json"
rng = np.random.default_rng(0); res = {}
for B in Bs:
    try:
        k, r = (pd.read_csv(f"{R}/an_{w}_B{B}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).set_index("step") for w in ("kp", "ref"))
    except FileNotFoundError:
        continue
    st = k.index.intersection(r.index); k, r = k.loc[st], r.loc[st]; t0 = int(st.min())
    e = {"start": t0, "n_eval": len(st)}
    import os                                                 # final = the scheduled end of the decay, both runs finished
    if int(st.max()) - t0 != B + 500 or not all(os.path.exists(f"{R}/an_{w}_B{B}_meta.json") for w in ("kp", "ref")):
        res[f"B{B}"] = {"status": f"incomplete (last common step {int(st.max())}, start {t0})"}; print(B, "incomplete"); continue
    for col in ("eval_loss", "eval_loss_b"):
        if col not in k: continue
        d = (k[col] - r[col]).values
        e[col] = {"final_diff": float(d[-3:].mean()), "pre_anneal_diff": float(np.interp(t0 + B, st, d)),
                  "final_kp": float(k[col].values[-3:].mean()), "final_ref": float(r[col].values[-3:].mean()),
                  "curve": [[int(a - t0), float(b)] for a, b in zip(st, d)]}
    try:
        dk, dr = (np.load(f"{R}/an_{w}_B{B}_detail.npz") for w in ("kp", "ref"))
        sk, sr = list(np.asarray(dk["step"]).astype(int)), list(np.asarray(dr["step"]).astype(int))
        common = [x for x in sk if x in sr][-3:]; ik, ir = [sk.index(x) for x in common], [sr.index(x) for x in common]
        for S in ("A", "B"):
            for nm, n in (("last", 1), ("last3", 3)):                  # each estimate with the bootstrap of its own evaluations
                dd = (dk[f"{S}_seq"][ik[-n:]] - dr[f"{S}_seq"][ir[-n:]]).mean(0); nb = len(dd) // 8; bl = dd[:nb * 8].reshape(nb, 8)
                bs = np.array([bl[rng.integers(0, nb, nb)].mean() for _ in range(1000)])
                e[f"{S}_{nm}_diff"] = float(dd.mean()); e[f"{S}_{nm}_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            e[f"{S}_last_steps"] = [int(x - t0) for x in common]
            if f"{S}_fb" in dk.files:
                e[f"{S}_fbin_last"] = ((dk[f"{S}_fb"][-1] - dr[f"{S}_fb"][-1]) / dk[f"{S}_fbcount"]).tolist()
    except FileNotFoundError:
        pass
    res[f"B{B}"] = e
    print(B, json.dumps({c: {q: v for q, v in e[c].items() if q != "curve"} if isinstance(e[c], dict) else e[c]
                         for c in e}), flush=True)
json.dump(res, open(out, "w"))
