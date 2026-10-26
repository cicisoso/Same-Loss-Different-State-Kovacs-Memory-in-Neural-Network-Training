# Figure S2: the matched loss split by target-token frequency. (a) Held-out gap at the match per frequency-rank bin, sets A
# and B; (b) lag and height of each bin's maximum after release; (c) excursion of each bin after release (set A, means over
# logarithmic bins of time); (d) gap at the end of identical anneals from matched checkpoints per bin (k_anneal_fbin.py,
# set A), with the gap of the same pairs at the match. With several k_detail files (independent seeds) panels a-c show the
# mean and its standard error.
# Usage: python figS2_tokens.py <k_detail.json>[,<k_detail.json>...] <out_stem> [k_anneal_fbin.json]
import sys, json, numpy as np
from figstyle import plt, C, W2, label, save
from matplotlib.ticker import FixedLocator, FuncFormatter

DS = [json.load(open(p)) for p in sys.argv[1].split(",")]; stem = sys.argv[2]; nrun = len(DS)
try:
    AF = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
except FileNotFoundError:
    AF = {}
logfmt = FuncFormatter(lambda v, _: f"{v:g}")
n = len(DS[0]["A"]["fbin_at_match"])
NAMES = [r"$\leq10$", r"$\leq10^2$", r"$\leq10^3$", r"$\leq10^4$", r"$>10^4$"][:n]
COLS = plt.cm.viridis(np.linspace(0.05, 0.85, n))
EDGES = 10 ** np.arange(0, 3.5, 0.1)                       # fixed logarithmic bins of steps after the match, shared by runs


def mse(v):
    """mean and standard error over runs (axis 0); no error bar for a single run"""
    v = np.asarray(v, float); m = v.mean(0)
    return m, (v.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else None)


def binned(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float); k = np.digitize(t, EDGES)
    return np.array([y[k == i].mean() if (k == i).any() else np.nan for i in range(1, len(EDGES))])


fig = plt.figure(figsize=(W2, 5.0))
gs = fig.add_gridspec(2, 2, wspace=0.62, hspace=0.62)
a, b, c, d = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
x = np.arange(n); wbar = 0.38

ext = []
for i, (S, col) in enumerate((("A", C["kov"]), ("B", C["teal"]))):
    m, se = mse([1e3 * np.array(D[S]["fbin_at_match"]) for D in DS])
    a.bar(x + (i - 0.5) * wbar, m, width=wbar, color=col, lw=0, label=f"set {S}", yerr=se,
          error_kw={"elinewidth": 0.8, "capsize": 0, "ecolor": C["ref"]})
    ext += list(m + (se if se is not None else 0)) + list(m - (se if se is not None else 0)) + [0.0]
lo, hi = min(ext), max(ext); rg = hi - lo                 # legend in a band above every bar and error bar
a.set_ylim(lo - 0.08 * rg, hi + 0.5 * rg)
a.axhline(0, color=C["grey"], lw=0.6); a.set_xticks(x); a.set_xticklabels(NAMES)
a.set_xlabel("frequency rank of target"); a.set_ylabel(r"gap at match ($10^{-3}$ nats)"); a.legend(loc="upper left", ncol=2)
label(a, "a")

tp, tse = mse([[st["t_peak"] for st in D["A"]["fbin_stats"]] for D in DS])
pk, pse = mse([[st["peak"] for st in D["A"]["fbin_stats"]] for D in DS])
b.errorbar(x, tp, yerr=tse, fmt="o-", color=C["kov"], ms=4, lw=1.0, elinewidth=0.8, capsize=0)
b.set_xticks(x); b.set_xticklabels(NAMES); b.set_xlabel("frequency rank of target")
b.set_ylabel("lag of maximum (steps)", color=C["kov"]); b.set_ylim(0, np.nanmax(tp + (tse if tse is not None else 0)) * 1.25)
b2 = b.twinx(); b2.errorbar(x, pk, yerr=pse, fmt="s--", color=C["gold"], ms=4, lw=1.0, elinewidth=0.8, capsize=0)
b2.set_ylabel("height (nats)", color=C["gold"]); b2.set_ylim(0, np.nanmax(pk + (pse if pse is not None else 0)) * 1.25)
b2.spines["right"].set_visible(True)
label(b, "b")

tc = np.sqrt(EDGES[:-1] * EDGES[1:])
for j in range(n):
    m, se = mse([binned(D["A"]["t"], D["A"]["fbin_curves"][j]) for D in DS]); ok = np.isfinite(m)
    c.plot(tc[ok], m[ok], color=COLS[j], lw=1.2, label=NAMES[j])
    if se is not None: c.fill_between(tc[ok], (m - se)[ok], (m + se)[ok], color=COLS[j], alpha=0.2, lw=0)
c.set_xscale("log"); c.set_xlim(1, 2500); c.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
c.xaxis.set_major_formatter(logfmt); c.axhline(0, color=C["grey"], lw=0.6)
c.set_xlabel("steps after match"); c.set_ylabel(r"$\Delta L$ per token (nats)")
c.legend(loc="upper right", title="frequency rank", title_fontsize=9, ncol=3, columnspacing=0.6, handlelength=1.0)
lo, hi = c.get_ylim(); c.set_ylim(lo, lo + (hi - lo) / 0.5)                   # headroom for the titled two-row legend
label(c, "c")

# (d) anneal gap per frequency bin: one line per annealed pair (transplant pair, B = 0, 200, 1000: coloured as in Figure 6;
# independent pairs: thin), with the gap of the transplant pair at the match (grey, dashed)
ACOL = {"B0": C["kov"], "B200": C["teal"], "B1000": C["cold"]}
for k_, e in AF.items():
    if "A_end" not in e: continue
    y = 1e3 * np.array(e["A_end"])
    if k_ in ACOL: d.plot(x, y, "o-", color=ACOL[k_], ms=3.5, lw=1.1, label=rf"anneal, $t_\mathrm{{d}}$ = {k_[1:]}", zorder=4)
    else: d.plot(x, y, "-", color=C["grey"], lw=0.8, alpha=0.8, zorder=3)
if "B0" in AF and "A_start" in AF["B0"]:
    d.plot(x, 1e3 * np.array(AF["B0"]["A_start"]), "--", color=C["grey"], lw=0.9)          # caption: dashed
d.axhline(0, color=C["grey"], lw=0.6); d.set_xticks(x); d.set_xticklabels(NAMES)
d.set_xlabel("frequency rank of target"); d.set_ylabel(r"gap per token ($10^{-3}$ nats)")
if d.get_legend_handles_labels()[0]:
    lo, hi = d.get_ylim(); d.set_ylim(lo, lo + (hi - lo) / 0.58); d.legend(loc="upper left")   # thin and dashed: caption
label(d, "d")

save(fig, stem, exclude_axes=[b2])
print("saved", stem, "runs", nrun)
