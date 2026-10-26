# Table-of-contents image (Advanced Science: 55 mm wide x 50 mm high). Top: the learning-rate histories of the Kovacs run
# (heat, cool until the held-out loss matches, release) and of the reference. Bottom: after the match the Kovacs run's loss
# rises above the reference's and returns, on identical minibatches.
# Usage: python toc_image.py <fig1_data.json> <out_stem>
import sys, json, numpy as np
from figstyle import plt, C

D = json.load(open(sys.argv[1])); stem = sys.argv[2]; P = D["pairs"][0]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
MM = 1 / 25.4


def clip_y(x, y, lo, hi):
    """points outside [lo, hi] become gaps, with the crossings of the limits inserted, so no path leaves the axes"""
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        cr = [(x[i - 1] + (L - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]), L) for L in (lo, hi)
              if (y[i - 1] - L) * (y[i] - L) < 0]                     # x increases, so crossings sort by x
        for xc, L in sorted(cr):
            xs.append(xc); ys.append(L)
        xs.append(x[i]); ys.append(y[i])
    xs, ys = np.array(xs, float), np.array(ys, float); ys[(ys < lo) | (ys > hi)] = np.nan
    return xs, ys


fig = plt.figure(figsize=(55 * MM, 50 * MM))
a = fig.add_axes([0.20, 0.70, 0.76, 0.22]); b = fig.add_axes([0.20, 0.17, 0.76, 0.40])

# learning-rate histories (steps relative to the match)
TM = P["TM"]; ph = P["phases"]
for s0, s1, lr in ph:
    col = C["hot"] if lr > 1.5e-3 else (C["cold"] if lr < 5e-4 else C["kov"])
    a.fill_between([s0 - TM, min(s1, TM + 600) - TM], 0, lr / 1e-3, color=col, lw=0, alpha=0.85)
a.plot([ph[0][0] - TM, 600], [1, 1], color=C["ref"], lw=0.9, ls="--")
a.set_xlim(ph[0][0] - TM, 600); a.set_ylim(0, 3.3); a.set_yticks([1, 3]); a.set_yticklabels(["1", "3"])
a.set_ylabel(r"$\eta/\eta_{\mathrm{ref}}$"); a.set_xticks([])
a.spines["bottom"].set_visible(False)
a.text(0.0, 1.02, "heat", color=C["hot"], transform=a.transAxes, fontsize=7, va="bottom")
a.text(0.27, 1.02, "cool to same loss", color=C["cold"], transform=a.transAxes, fontsize=7, va="bottom")
a.text(1.0, 1.02, "release", color=C["kov"], transform=a.transAxes, fontsize=7, va="bottom", ha="right")

# excursion after the match
t = np.array(P["t"]); y = np.array(P["dA"]); m = t <= 600; post = m & (t >= 0)
ys = np.convolve(np.r_[y[0], y, y[-1]], np.ones(3) / 3, "valid")
b.axhline(0, color=C["ref"], lw=0.9, ls="--")
b.fill_between(t[post], 0, 1e3 * ys[post], where=ys[post] > 0, color=C["kov"], alpha=0.18, lw=0)
b.plot(*clip_y(t[m], 1e3 * ys[m], -12, 80), color=C["kov"], lw=1.4)
b.plot([0], [0], "o", ms=3.5, color="black", zorder=5)
b.set_xlim(ph[0][0] - TM, 600); b.set_ylim(-12, 80)
b.set_xlabel("steps after match"); b.set_ylabel(r"$\Delta L$ ($10^{-3}$)")
b.text(0.97, 0.88, "memory", color=C["kov"], transform=b.transAxes, ha="right", fontsize=8, fontweight="bold")
fig.savefig(f"{stem}.pdf"); fig.savefig(f"{stem}.png", dpi=600); fig.savefig(f"{stem}.tif", dpi=600)
print("saved", stem)
