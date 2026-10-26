# E10 quick-look figure: loss curves and loss differences to the reference for every run of one family.
# Usage: python -I k_plot.py <results_dir> <ref_tag> <tags,comma> <out.png>
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

R, ref_tag, tags, out = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
ref = pd.read_csv(f"{R}/{ref_tag}_eval.tsv", sep="\t").dropna(subset=["eval_loss"])
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(ref.step, ref.eval_loss, "k-", lw=1, label=ref_tag)
for tag in tags:
    try:
        d = pd.read_csv(f"{R}/{tag}_eval.tsv", sep="\t").dropna(subset=["eval_loss"])
    except FileNotFoundError:
        continue
    m = json.load(open(f"{R}/{tag}_meta.json")) if __import__("os").path.exists(f"{R}/{tag}_meta.json") else {"phases": []}
    rel = next((p["matched_at"] for p in m["phases"] if p.get("matched_at")), None)
    j = d[(d.step > ref.step.min()) & (d.step <= ref.step.max())].copy()
    for c in ("eval_loss", "wnorm"): j[c + "_ref"] = np.interp(j.step, ref.step, ref[c])
    l, = ax[0].plot(d.step, d.eval_loss, lw=1, label=tag)
    ax[1].plot(j.step, j.eval_loss - j.eval_loss_ref, lw=1, color=l.get_color(), label=tag)
    ax[2].plot(j.step, j.wnorm - j.wnorm_ref, lw=1, color=l.get_color(), label=tag)
    if rel:
        for k in (1, 2): ax[k].axvline(rel, color=l.get_color(), ls=":", lw=0.8)
ax[0].set_ylim(ref.eval_loss.min() - 0.15, ref.eval_loss.min() + 0.35); ax[0].set_ylabel("held-out loss")
ax[1].axhline(0, color="k", lw=0.5); ax[1].set_ylabel("loss - reference"); ax[1].set_ylim(-0.08, 0.08)
ax[2].axhline(0, color="k", lw=0.5); ax[2].set_ylabel("weight norm - reference")
for k in range(3): ax[k].set_xlabel("step"); ax[k].legend(fontsize=7)
plt.tight_layout(); plt.savefig(out, dpi=110)
