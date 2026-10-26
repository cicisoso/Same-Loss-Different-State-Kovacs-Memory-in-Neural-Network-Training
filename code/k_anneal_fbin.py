# E10 exploratory: the anneal gap split by target-token frequency. For each annealed matched pair, the per-token held-out
# loss difference (Kovacs minus reference) in the five unigram frequency-rank bins of the detail logs, at the start of the
# anneal pair (the match) and at its last evaluation, on sets A and B. The bins are aggregated over all sequences, so no
# interval is given (descriptive only).
# Usage: python -I k_anneal_fbin.py <results_dir> <name=kov:ref,...> <out.json>
import sys, json, numpy as np

R = sys.argv[1]; specs = [x.split("=") for x in sys.argv[2].split(",") if x]; out = sys.argv[3]
res = {}
for name, spec in specs:
    a_, b_ = spec.split(":")
    try:
        da, db = np.load(f"{R}/{a_}_detail.npz"), np.load(f"{R}/{b_}_detail.npz")
    except FileNotFoundError as e:
        print(name, "missing", e.filename); continue
    sa, sb = list(da["step"].astype(int)), list(db["step"].astype(int)); common = [s for s in sa if s in sb]
    if not common:
        print(name, "no common evaluations"); continue
    e = {"pair": [a_, b_], "edges": da["fb_edges"].tolist(), "start": int(common[0]), "end": int(common[-1])}
    for S in ("A", "B"):
        if f"{S}_fb" not in da.files: continue
        cnt = da[f"{S}_fbcount"]; e[f"{S}_weight"] = (cnt / cnt.sum()).tolist()
        for lab, s in (("start", common[0]), ("end", common[-1])):
            d = (da[f"{S}_fb"][sa.index(s)] - db[f"{S}_fb"][sb.index(s)]) / cnt
            e[f"{S}_{lab}"] = d.tolist(); e[f"{S}_{lab}_total"] = float((d * cnt).sum() / cnt.sum())
    res[name] = e
    print(name, "A start", np.round(e.get("A_start", []), 4).tolist(), "end", np.round(e.get("A_end", []), 4).tolist(),
          "total", round(e.get("A_end_total", np.nan), 4), flush=True)
json.dump(res, open(out, "w"))
