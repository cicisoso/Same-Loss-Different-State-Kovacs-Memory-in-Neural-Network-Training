# E10 exploratory (PREREG addendum 2026-10-20 09:44): anneal gaps after the function-preserving norm changes. With Y(x) the
# held-out loss at the end of the B = 0 anneal of checkpoint x (identical schedule and minibatches):
#   G   = Y(an_kp_B0)   - Y(an_ref_B0)   the observed anneal gap
#   G_N = Y(an_kpN_B0)  - Y(an_ref_B0)   Kovacs network with both norm reductions
#   H_N = Y(an_refN_B0) - Y(an_ref_B0)   reference with both norm increases (= C_R)
#   C_K = Y(an_kpN_B0)  - Y(an_kp_B0)    = G_N - G, bootstrapped directly on the shared sequences (Codex round 4)
# Each as the last-evaluation difference and the mean over the last three evaluations, on sets A and B, with a block
# bootstrap over validation sequences (blocks of 8, 1000 resamples) of the same evaluations, as in k_anneal.py; plus the
# held-out-loss trajectories of the differences.
# Usage: python -I k_annealN.py <results_dir> [out.json]
import sys, json, numpy as np, pandas as pd

R = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else f"{R}/k_annealN.json"
rng = np.random.default_rng(0); res = {}
for nm, a_, b_ in (("G", "an_kp_B0", "an_ref_B0"), ("G_N", "an_kpN_B0", "an_ref_B0"), ("H_N", "an_refN_B0", "an_ref_B0"),
                   ("C_K", "an_kpN_B0", "an_kp_B0")):
    try:
        k, r = (pd.read_csv(f"{R}/{t}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).set_index("step") for t in (a_, b_))
        dk, dr = (np.load(f"{R}/{t}_detail.npz") for t in (a_, b_))
    except FileNotFoundError:
        continue
    st = k.index.intersection(r.index); t0 = int(st.min()); e = {"pair": [a_, b_], "start": t0}
    import os                                                 # final = the scheduled end of the decay, both runs finished
    if int(st.max()) - t0 != 500 or not all(os.path.exists(f"{R}/{t}_meta.json") for t in (a_, b_)):
        res[nm] = {"status": f"incomplete (last common step {int(st.max())}, start {t0})"}; print(nm, res[nm]["status"]); continue
    for col in ("eval_loss", "eval_loss_b"):
        d = (k.loc[st, col] - r.loc[st, col]).values
        e[col] = {"first_diff": float(d[0]), "final_diff": float(d[-3:].mean()), "curve": [[int(x - t0), float(y)] for x, y in zip(st, d)]}
    sk, sr = list(np.asarray(dk["step"]).astype(int)), list(np.asarray(dr["step"]).astype(int))
    common = [x for x in sk if x in sr][-3:]; ik, ir = [sk.index(x) for x in common], [sr.index(x) for x in common]
    for S in ("A", "B"):
        for lab, n in (("last", 1), ("last3", 3)):
            dd = (dk[f"{S}_seq"][ik[-n:]] - dr[f"{S}_seq"][ir[-n:]]).mean(0); nb = len(dd) // 8; bl = dd[:nb * 8].reshape(nb, 8)
            bs = np.array([bl[rng.integers(0, nb, nb)].mean() for _ in range(1000)])
            e[f"{S}_{lab}_diff"] = float(dd.mean()); e[f"{S}_{lab}_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    res[nm] = e
    print(nm, json.dumps({q: (round(v, 5) if isinstance(v, float) else [round(x, 5) for x in v] if isinstance(v, list) and q.endswith("ci") else v)
                         for q, v in e.items() if q not in ("eval_loss", "eval_loss_b", "pair")}),
          "first", round(e["eval_loss"]["first_diff"], 5), flush=True)
json.dump(res, open(out, "w"))
