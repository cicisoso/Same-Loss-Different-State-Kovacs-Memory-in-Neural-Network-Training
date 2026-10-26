# Figure 5: breadth and sign. (a) heat->cool Kovacs excursions of language models (web text at two sizes, code, PubMed);
# (b) the cool->heat protocol on the same models; (c) MovieLens matrix factorisation and (e) the CIFAR-10 CNN with batch
# norm, each with momentum and plain SGD, relative to the reference loss at the match (separate panels: the two differ in
# size by more than an order of magnitude); (d) signed extremum of every system and protocol; (f) effect of exchanging the
# filter norms of the CNN in the Kovacs and in the reference background, and their interaction (2x2 exchange; the
# continuations themselves are in Figure S1d). Three rows of two panels at double-column width.
# Usage: python fig5_breadth.py <fig5_data.json> <out_stem>
import sys, json, numpy as np
from figstyle import plt, C, W2, label, save
from matplotlib.ticker import FixedLocator, FuncFormatter

D = json.load(open(sys.argv[1])); stem = sys.argv[2]; S = D["systems"]
logfmt = FuncFormatter(lambda v, _: f"{v:g}")
LM = (("web", "web 30M", C["kov"], "-"), ("web124", "web 124M", C["ref"], "-"), ("code", "code", C["teal"], "-"),
      ("pubmed", "PubMed", C["gold"], "-"), ("wd0", "web, no decay", C["grey"], "--"))
OTHER = {"c": (("mfn", "momentum", C["green"]), ("mf0n", "plain SGD", C["green"])),
         "e": (("cifar", "momentum", C["hot"]), ("cifar0", "plain SGD", C["hot"]))}
SHORT = {"web": "web 30M", "web124": "web 124M", "code": "code", "pubmed": "PubMed", "wd0": "web, no decay",
         "mfn": "MovieLens", "mf0n": "MovieLens, SGD", "cifar": "CIFAR", "cifar0": "CIFAR, SGD"}
TICK = {"web": "web 30M", "web124": "web 124M", "code": "code", "pubmed": "PubMed", "wd0": "no decay",   # panel d rows
        "mfn": "MF", "mf0n": "MF, SGD", "cifar": "CNN", "cifar0": "CNN, SGD"}


def smooth(y, k=3):
    y = np.asarray(y, float); return np.convolve(np.r_[[y[0]] * (k // 2), y, [y[-1]] * (k // 2)], np.ones(k) / k, "valid")


def logbin(t, y, per_decade=12):
    """display only: means over logarithmically spaced bins of steps after the match (statistics use the raw evaluations)"""
    t, y = np.asarray(t, float), np.asarray(y, float); m = t >= 1; t, y = t[m], y[m]
    edges = 10 ** np.arange(0, np.log10(t.max()) + 1 / per_decade, 1 / per_decade); k = np.digitize(t, edges)
    tb = np.array([np.exp(np.log(t[k == i]).mean()) for i in np.unique(k)]); yb = np.array([y[k == i].mean() for i in np.unique(k)])
    return tb, yb


def logx(ax, hi=2500):
    ax.set_xscale("log"); ax.set_xlim(1, hi); ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(logfmt); ax.axhline(0, color=C["grey"], lw=0.6)


def curve(ax, e, col, ls="-", lw=1.2, rel=False, per_decade=12, **kw):
    t, y = logbin(e["t"], e["dA"], per_decade)
    if rel: y = 100 * y / e["ref_loss_at_match"]
    ax.plot(t, y, color=col, ls=ls, lw=lw, **kw)


fig, axs = plt.subplots(3, 2, figsize=(W2, 7.6), gridspec_kw={"wspace": 0.38, "hspace": 0.62})
a, b, c, d, e, f = axs.ravel()


def headroom(ax, frac):
    """raise the upper y limit so the data fill the lower `frac` of the axes, leaving the top for the legend"""
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, lo + (hi - lo) / frac)


# (a), (b) language models, both protocols (one legend, in a)
for ax, prot in ((a, "kp"), (b, "km")):
    for key, nm, col, ls in LM:
        if key in S and prot in S[key]: curve(ax, S[key][prot], col, ls, label=nm)
    logx(ax); ax.set_xlabel("steps after match"); ax.set_ylabel(r"$\Delta L$ (nats)")
headroom(a, 0.5)                                             # five systems: two legend columns above the 124M maximum
a.set_yticks([0, 0.05, 0.1])
if a.get_legend_handles_labels()[0]: a.legend(loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.2)
label(a, "a"); label(b, "b")

# (c) matrix factorisation and (e) the CNN, relative to the reference loss at the match; solid heat->cool, dashed cool->heat
for ax, pan in ((c, "c"), (e, "e")):
    for key, nm, col in OTHER[pan]:
        if key not in S: continue
        for prot, ls in (("kp", "-"), ("km", "--")):
            if prot in S[key]:
                curve(ax, S[key][prot], col, ls, lw=1.3 if nm == "momentum" else 0.7, rel=True,
                      per_decade=6 if pan == "e" else 12, label=nm if prot == "kp" else None)
    logx(ax); ax.set_xlabel("steps after match"); ax.set_ylabel(r"$\Delta L/L_{\mathrm{ref}}$ (%)")
    headroom(ax, 0.62)
    if ax.get_legend_handles_labels()[0]:                    # MovieLens: upper left, clear of the hump at 100-300 steps
        ax.legend(loc="upper left" if pan == "c" else "upper right", title="MovieLens" if pan == "c" else "CIFAR-10")
label(c, "c"); label(e, "e")

# (d) signed extremum of every system, relative to the reference loss at the match
keys = [k for k in SHORT if k in S]; y = np.arange(len(keys))[::-1]
for yi, k in zip(y, keys):
    for prot, mk in (("kp", "o"), ("km", "s")):
        if prot not in S[k]: continue
        v = 100 * S[k][prot]["stats"]["ext"] / S[k][prot]["ref_loss_at_match"]
        col = C["hot"] if prot == "kp" else C["cold"]
        d.plot([v], [yi], mk, ms=4.5, mfc=col if prot == "kp" else "white", mec=col, mew=1.0)
d.set_yticks(y); d.set_yticklabels([TICK[k] for k in keys], fontsize=0.9 * plt.rcParams["ytick.labelsize"])   # nine rows
d.set_xlabel("signed extremum (% of loss)"); d.set_ylim(-0.7, len(keys) - 0.3)  # legend in the empty upper right
d.plot([0, 0], [-0.7, len(keys) - 0.4], color=C["grey"], lw=0.6, zorder=0)   # zero line over the data rows only
from matplotlib.lines import Line2D
d.legend(handles=[Line2D([], [], ls="", marker="o", ms=4.5, color=C["hot"], label="heat, cool"),
                  Line2D([], [], ls="", marker="s", ms=4.5, mfc="white", mec=C["cold"], label="cool, heat")],
         loc="upper right", ncol=1, handletextpad=0.1)
label(d, "d")

# (f) CIFAR factorial: effect of the filter norms in the Kovacs background (Kovacs network at its own vs the reference's
# norms) and in the reference background (reference at the Kovacs vs its own norms), and their difference, the interaction;
# each as a percentage of the observed excursion (area over 0-300 steps, mean offset over 1000-2000 steps)
F = D.get("cifar_factorial")
if F:
    x = np.arange(2); wbar = 0.26
    for k, (key, nm, col) in enumerate((("norm_K", "norms, Kovacs background", C["hot"]),
                                        ("norm_R", "norms, ref. background", C["hot_soft"]),
                                        ("interaction", "interaction", C["grey"]))):
        v = [100 * F[g][key] / F[g]["KK"] for g in ("area", "late")]
        f.bar(x + (k - 1) * wbar, v, width=wbar, color=col, lw=0, label=nm)
    f.set_xticks(x); f.set_xticklabels(["0\u2013300", "1000\u20132000"]); f.set_xlabel("steps after match")
    f.axhline(0, color=C["grey"], lw=0.6); f.legend(loc="upper right", ncol=1)
    headroom(f, 0.55)
f.set_ylabel("% of excursion")
label(f, "f")

save(fig, stem)
print("saved", stem)
