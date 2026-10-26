# Figure 2: Kovacs positivity. (a) mode displacement D(r) at the match for the GPT protocol (p = q = 1); (b) two-mode
# decomposition of the excursion; (c) exact SGD model, excursion vs curvature ratio kappa (exact recursion vs linear
# theory); (d) exact SGD model, excursion vs gradient-noise strength 1/B.
# Usage: python -I fig2_theory.py <results_dir> <out_stem>
import sys, json, numpy as np
import matplotlib.ticker as mticker
from figstyle import plt, C, W2, label, save

R, stem = sys.argv[1], sys.argv[2]
fig, axs = plt.subplots(2, 3, figsize=(W2, 5.3), gridspec_kw={"wspace": 0.62, "hspace": 0.55})
axs = axs.ravel()

# (a) D(r) for heat 3x for 300 steps, cool 1/8 for 704 steps (intrinsic s_a = 900, s_c = 88 reference steps)
gh, g0, gl, sa, sc = 3.0, 1.0, 0.125, 900.0, 88.0
r = np.logspace(-4.5, 0.5, 600)
D = (gh - gl) * np.exp(-r * sc) - (gh - g0) * np.exp(-r * (sc + sa)) - (g0 - gl)
ax = axs[0]
ax.fill_between(r, 0, D, where=D > 0, color=C["hot_soft"], lw=0); ax.fill_between(r, 0, D, where=D < 0, color=C["cold_soft"], lw=0)
ax.plot(r, D, color="k", lw=1.2); ax.axhline(0, color=C["grey"], lw=0.6)
rs = r[np.nonzero(np.diff(np.sign(D[r > 1e-4])))[0][0] + np.argmax(r > 1e-4)]
ax.set_xscale("log"); ax.set_xlabel(r"mode rate $r$ (per step)"); ax.set_ylabel(r"displacement $D(r)$")
ax.text(1.0e-3, 1.42, "slow: hot", color=C["hot"], ha="center", va="bottom")
ax.text(0.11, -1.12, "fast: cold", color=C["cold"], ha="center", va="top")
ax.annotate(r"$r^*$", (rs, 0), (rs * 4.0, 0.6), arrowprops={"arrowstyle": "-", "lw": 0.6})
ax.xaxis.set_major_locator(mticker.FixedLocator([1e-4, 1e-2, 1]))
ax.set_ylim(-1.9, 2.2); ax.set_xlim(3e-5, 3); label(ax, "a")

# (b) two modes, equal weights at r_f = 0.1 and r_s = 0.004: contributions after release and their sum
rf, rsl = 0.1, 0.004
Df, Ds = [(gh - gl) * np.exp(-q * sc) - (gh - g0) * np.exp(-q * (sc + sa)) - (g0 - gl) for q in (rf, rsl)]
af, as_ = 1.0, -Df / Ds                                               # weights that make the two modes cancel at release
sg = np.logspace(0, 3.5, 400)
cf, cs = af * Df * np.exp(-rf * sg), as_ * Ds * np.exp(-rsl * sg)
ax = axs[1]
ax.plot(sg, cs, color=C["hot"], lw=1.2); ax.plot(sg, cf, color=C["cold"], lw=1.2)
ax.plot(sg, cf + cs, color="k", lw=1.8); ax.axhline(0, color=C["grey"], lw=0.6)
ax.set_xscale("log"); ax.set_xlabel(r"steps after release $\sigma$"); ax.set_ylabel(r"$\Delta E\,/\,g_0$")
ax.xaxis.set_major_locator(mticker.FixedLocator([1, 10, 100, 1000]))
ax.text(1.3, 0.98, "slow", color=C["hot"], va="bottom")
ax.text(1.3, -0.97, "fast", color=C["cold"], va="top", ha="left")
ax.text(300, 0.42, "sum", color="k", va="bottom", ha="left")
ax.set_ylim(-1.3, 1.3); label(ax, "b")

# (c) exact recursion vs linear theory
kap = json.load(open(f"{R}/ts_kappa.json")); ks = np.array(sorted(map(int, kap)))
ex = np.array([kap[str(k)]["exact"]["hump_rel"] for k in ks]); li = np.array([kap[str(k)]["linear"]["hump_rel"] for k in ks])
ax = axs[2]
ax.plot(ks, li, color=C["grey"], lw=1.3); ax.plot(ks, ex, "o", color=C["kov"], ms=4.5)
ax.set_xscale("log", base=2); ax.set_xlabel(r"curvature ratio $\kappa$"); ax.set_ylabel("hump / heating excess")
ax.xaxis.set_major_locator(mticker.FixedLocator([1, 8, 64, 512]))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.text(2 ** 5.6, 0.25, "linear", color=C["ref"], ha="right")
ax.text(2 ** 5.0, 0.105, "exact", color=C["kov"], ha="left", va="top")
ax.set_ylim(-0.01, 0.3); label(ax, "c")

# (d) peak-time law in the exact model: observed peak step against the prediction from the protocol and nu
pk = json.load(open(f"{R}/ts_peak.json"))
cmap = {0.2: C["teal"], 0.333: C["kov"], 0.5: C["hot"], 0.667: C["gold"]}
col = lambda nu: cmap[min(cmap, key=lambda k: abs(k - nu))]
ax = axs[3]
for r_ in pk:
    ax.plot(r_["pred"], r_["t_peak"], "o", color=col(r_["nu"]), ms=4.5, mew=0)
ax.plot([2, 40], [2, 40], color=C["grey"], lw=0.8, ls="--")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2, 40); ax.set_ylim(2, 40)
for a_ in (ax.xaxis, ax.yaxis):
    a_.set_major_locator(mticker.FixedLocator([2, 5, 10, 20, 40])); a_.set_major_formatter(mticker.ScalarFormatter())
    a_.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel(r"predicted lag $\sigma^*$ (steps)"); ax.set_ylabel("observed lag (steps)")
for k_, (nu_, c_) in enumerate(sorted(cmap.items())):
    ax.text(0.97, 0.05 + 0.12 * (3 - k_), rf"$\nu$ = {nu_:.2g}", transform=ax.transAxes, color=c_, ha="right", va="bottom")
label(ax, "d")

# (e) the protocol sets the timescale: sigma*/s_c against the level ratio Q, theory curves for each nu
ax = axs[4]
Q = np.linspace(1.4, 4.0, 100)
for nu_, c_ in sorted(cmap.items()):
    ax.plot(Q, 1 / (Q ** (1 / (1 + nu_)) - 1), color=c_, lw=1.2)
for r_ in pk:
    gh, gl = r_["eta_h"] / 0.5, r_["eta_l"] / 0.5
    if r_["s_c"] >= 10: ax.plot((gh - gl) / (1 - gl), r_["t_peak"] / r_["s_c"], "o", color=col(r_["nu"]), ms=4.5, mew=0)
ax.plot(Q, 1 / (Q - 1), color=C["ref"], lw=1.2, ls=":")
ax.text(2.0, 1 / (2.0 - 1) * 0.55, r"$\nu$ = 0", color=C["ref"], ha="right", va="top")
ax.set_yscale("log"); ax.set_xlabel(r"level ratio $Q$"); ax.set_ylabel(r"$\sigma^*/s_c$")
ax.yaxis.set_major_locator(mticker.FixedLocator([0.3, 0.5, 1, 2, 4])); ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mticker.NullFormatter()); ax.set_ylim(0.28, 5)
label(ax, "e")

# (d) batch size: the hump is a fixed fraction of the noise-driven heating excess until that excess no longer beats the
# drift gain of the hotter step (B = 256); full-batch gradient descent has no Kovacs protocol (heating lowers the loss)
nz = json.load(open(f"{R}/ts_noise.json"))
Bs = np.array([4, 16, 64, 256]); hp = np.array([nz[f"B_{b}"]["hump"] for b in Bs]); hx = np.array([nz[f"B_{b}"]["heat_excess"] for b in Bs])
ax = axs[5]
ok = hp > 1e-6
xx = np.array([5e-5, 0.1]); ax.plot(xx, 0.12 * xx, color=C["grey"], lw=0.8, ls="--")
ax.plot(hx[ok], hp[ok], "o", color=C["kov"], ms=5)
ax.plot(hx[~ok], np.full((~ok).sum(), 1.5e-6), "x", color=C["kov"], ms=5.5, mew=1.2)
for b, x_, y_ in zip(Bs, hx, np.where(ok, hp, 1.5e-6)):
    if y_ > 1e-5: ax.text(x_ * 1.5, y_ * 0.7, f"{b}", color=C["ref"], ha="left", va="top")
    else: ax.text(x_ * 1.6, y_, f"{b}", color=C["ref"], ha="left", va="center")
ax.text(0.04, 0.96, "batch size", transform=ax.transAxes, color=C["ref"], ha="left", va="top")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("heating excess"); ax.set_ylabel("hump")
ax.xaxis.set_major_locator(mticker.FixedLocator([1e-4, 1e-3, 1e-2, 1e-1])); ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.yaxis.set_major_locator(mticker.FixedLocator([1e-6, 1e-4, 1e-2])); ax.yaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_ylim(8e-7, 2e-2); ax.set_xlim(6e-5, 0.2); label(ax, "f")
for a in axs: a.tick_params(which="both", direction="out")
save(fig, stem)
print("saved", stem, "r*", rs, "two-mode weights", af, as_)
