# Figure 1: the Kovacs protocol in GPT training. (a) learning-rate protocols; (b) held-out loss of both runs; (c) the same
# around the match; (d) post-release excursion of the flagship pair on the matched set A, the untouched set B and the
# per-step training loss on the shared minibatches; (e) excursions of independent seeds (mean +- s.e.m.; exploratory pairs as
# grey dashed lines); (f) per-seed peak excursion on A and B (paired).
# Usage: python fig1_phenom.py <fig1_data.json> <out_stem> [n_flagship=0]
import sys, json, numpy as np
from figstyle import plt, C, W2, label, save
from matplotlib.ticker import FixedLocator, FuncFormatter

D = json.load(open(sys.argv[1])); stem = sys.argv[2]
P = D["pairs"]; F = P[int(sys.argv[3]) if len(sys.argv) > 3 else 0]
logfmt = FuncFormatter(lambda v, _: f"{v:g}")


def smooth(y, k=3):
    y = np.asarray(y, float); return np.convolve(np.r_[[y[0]] * (k // 2), y, [y[-1]] * (k // 2)], np.ones(k) / k, "valid")


fig, axs = plt.subplots(2, 3, figsize=(W2, 5.3), gridspec_kw={"wspace": 0.62, "hspace": 0.55})
ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axs.ravel()
TM = F["TM"]; t0 = F["phases"][0][0]; tm = TM - t0

# (a) protocols
x, yk = [0], [1.0]
for a, b, lr in F["phases"]:
    x += [a - t0, b - t0]; yk += [lr * 1e3, lr * 1e3]
xa = np.minimum(np.array(x, float), 2500)
ax_a.plot([0, 2500], [1, 1], color=C["ref"], lw=1.2, ls="--"); ax_a.plot(xa, yk, color=C["kov"], lw=1.6)
ax_a.set_yscale("log"); ax_a.set_ylim(0.045, 6); ax_a.set_xlim(-100, 2500)
ax_a.yaxis.set_major_locator(FixedLocator([0.125, 1, 3])); ax_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: {0.125: "1/8"}.get(v, f"{v:g}")))
ax_a.yaxis.set_minor_locator(FixedLocator([]))
ax_a.set_xlabel("steps after checkpoint"); ax_a.set_ylabel(r"$\eta\,/\,\eta_{\rm ref}$")
ax_a.text(340, 3.0, "heat", ha="left", va="center", color=C["hot"])
ax_a.text((300 + tm) / 2, 0.16, "cool", ha="center", va="bottom", color=C["cold"])
ax_a.text(tm + 60, 1.25, "release", ha="left", va="bottom", color="k")
label(ax_a, "a")

# (b) absolute held-out loss
A = F["abs"]
rs, rl = np.array(A["ref_step"]) - t0, np.array(A["ref_loss"]); ks, kl = np.array(A["kov_step"]) - t0, np.array(A["kov_loss"])
w2, wk2 = rs <= 2500, ks <= 2500
ax_b.plot(rs[w2], rl[w2], color=C["ref"], lw=1.0); ax_b.plot(ks[wk2], kl[wk2], color=C["kov"], lw=1.0)
lo = min(rl[w2].min(), kl[wk2].min()); ax_b.set_xlim(-100, 2500); ax_b.set_ylim(lo - 0.1, kl.max() + 0.05)
ax_b.set_xlabel("steps after checkpoint"); ax_b.set_ylabel("held-out loss (nats)")
ax_b.text(420, kl.max() - 0.05, "Kovacs", color=C["kov"], va="top")
ax_b.text(1300, np.interp(1300, rs, rl) + 0.07, "reference", color=C["ref"], va="bottom")
ax_b.axvline(tm, ymax=0.3, color=C["grey"], lw=0.8, ls=":")
label(ax_b, "b")

# (c) around the match
wr = (rs >= tm - 60) & (rs <= tm + 420); wk = (ks >= tm - 60) & (ks <= tm + 420)
ax_c.plot(rs[wr] - tm, rl[wr], color=C["ref"], lw=1.0); ax_c.plot(ks[wk] - tm, kl[wk], color=C["kov"], lw=1.2)
ax_c.axvline(0, color=C["grey"], lw=0.8, ls=":"); ax_c.set_xlim(-60, 420)
ax_c.set_xlabel("steps after match"); ax_c.set_ylabel("held-out loss (nats)")
label(ax_c, "c")

# (d) flagship excursion: A, B, training loss (log time)
t = np.array(F["t"], float); dA = np.array(F["dA"]); m = t >= 1
ax_d.axhline(0, color=C["grey"], lw=0.6)
if "dTrain" in F:
    tt, dt = np.array(F["train_t"], float), np.array(F["dTrain"]); k = tt >= 1
    ax_d.plot(tt[k], smooth(dt[k], 9), color=C["light"], lw=1.0, label="train")
if "dB" in F:
    ax_d.plot(t[m], smooth(np.array(F["dB"])[m]), color=C["teal"], lw=1.2, label="set B")
ax_d.plot(t[m], smooth(dA[m]), color=C["kov"], lw=1.6, label="set A")
ax_d.set_xscale("log"); ax_d.set_xlim(1, 2500); ax_d.set_xlabel("steps after release")
ax_d.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000])); ax_d.xaxis.set_major_formatter(logfmt)
ax_d.set_ylim(-0.015, 1.9 * max(smooth(dA[m]).max(), 0.01)); ax_d.set_ylabel(r"$\Delta L$ (nats)")   # legend above the data
ax_d.legend(loc="upper right")
label(ax_d, "d")

# (e) independent seeds (pairs with their own reference); mean +- s.e.m. on a common grid
grid = np.r_[np.arange(0, 100, 5), np.arange(100, 300, 10), np.arange(300, 2001, 50)]
curves = []
for p in P:
    tp, dp = np.array(p["t"], float), np.array(p["dA"]); k = tp >= 0
    if tp.max() < 1500 or not p.get("seed", True): continue
    curves.append(np.interp(grid, tp[k], smooth(dp[k])))
for p in P:                                                     # exploratory pairs (first network): grey, not averaged
    tp, dp = np.array(p["t"], float), np.array(p["dA"]); k = tp >= 0
    if p.get("seed", True) or tp.max() < 1500: continue
    gm = grid <= tp.max(); ax_e.plot(grid[gm], np.interp(grid[gm], tp[k], smooth(dp[k])), color=C["grey"], lw=0.9, ls="--")
if curves:
    cv = np.array(curves)
    for cu in cv: ax_e.plot(grid, cu, color=C["kov"], lw=0.6, alpha=0.35)
    mu = cv.mean(0); se = cv.std(0, ddof=1) / np.sqrt(len(cv)) if len(cv) > 1 else 0 * mu
    ax_e.fill_between(grid, mu - se, mu + se, color=C["kov"], alpha=0.25, lw=0); ax_e.plot(grid, mu, color=C["kov"], lw=1.6)
    ax_e.text(0.97, 0.95, f"n = {len(cv)}", transform=ax_e.transAxes, ha="right", va="top")
y0, y1 = ax_e.get_ylim(); ax_e.set_ylim(y0, y1 + 0.18 * (y1 - y0))   # headroom: the n label stays above the band
ax_e.axhline(0, color=C["grey"], lw=0.6)
ax_e.set_xlim(0, 2000); ax_e.xaxis.set_major_locator(FixedLocator([0, 1000, 2000]))
ax_e.set_xlabel("steps after release"); ax_e.set_ylabel(r"$\Delta L$ (nats)")
label(ax_e, "e")

# (f) peak (3-evaluation running mean, 10-150 steps after release) per pair on A and B
pk = []
for p in P:
    if not p.get("seed", True): continue
    tp = np.array(p["t"], float); w = (tp >= 10) & (tp <= 150)
    if not w.any(): continue
    a_ = smooth(np.array(p["dA"]))[w].max(); b_ = smooth(np.array(p["dB"]))[w].max() if "dB" in p else np.nan
    pk.append((a_, b_))
pk = np.array(pk) if pk else np.zeros((0, 2))
for a_, b_ in pk:
    if np.isfinite(b_): ax_f.plot([0, 1], [a_, b_], color=C["light"], lw=1.0)
ax_f.plot(np.zeros(len(pk)), pk[:, 0], "o", color=C["kov"], ms=5); ax_f.plot(np.ones(len(pk)), pk[:, 1], "o", color=C["teal"], ms=5)
ax_f.set_xticks([0, 1]); ax_f.set_xticklabels(["set A", "set B"]); ax_f.set_xlim(-0.5, 1.5)
ax_f.set_ylim(0, max(0.01, np.nanmax(pk) * 1.25) if len(pk) else 0.1); ax_f.set_ylabel("peak excursion (nats)")
label(ax_f, "f")
save(fig, stem)
print("saved", stem, "pairs", len(P), "seed curves", len(curves), "peaks", pk.round(4).tolist())
