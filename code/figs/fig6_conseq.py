# Figure 6: matched checkpoints are not interchangeable. (a) Loss difference between the Kovacs and reference checkpoints
# (same held-out loss) through identical anneals started after B further steps; (b) final differences on sets A and B with
# block-bootstrap 95% intervals, against the prediction of a clock shared by both networks; (c) independent matched pairs annealed from their match, against
# their own free excursion and shared-clock prediction; (d) the excursion
# from a checkpoint 4000 steps older. Two rows of two panels at double-column width (the token-frequency split is in the
# Supporting Information, Figure S2).
# Usage: python fig6_conseq.py <fig6_data.json> <out_stem>
import sys, json, numpy as np
from figstyle import plt, C, W2, label, save
from matplotlib.ticker import FixedLocator, FuncFormatter

D = json.load(open(sys.argv[1])); stem = sys.argv[2]
logfmt = FuncFormatter(lambda v, _: f"{v:g}")


def logbin(t, y, per_decade=12):
    t, y = np.asarray(t, float), np.asarray(y, float); m = t >= 1; t, y = t[m], y[m]
    edges = 10 ** np.arange(0, np.log10(t.max()) + 1 / per_decade, 1 / per_decade); k = np.digitize(t, edges)
    u = np.unique(k); return np.array([np.exp(np.log(t[k == i]).mean()) for i in u]), np.array([y[k == i].mean() for i in u])


def logx(ax, hi=2500):
    ax.set_xscale("log"); ax.set_xlim(1, hi); ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(logfmt); ax.axhline(0, color=C["grey"], lw=0.6)


fig, axs = plt.subplots(2, 2, figsize=(W2, 4.9), gridspec_kw={"wspace": 0.42, "hspace": 0.55})
a, b, c, d = axs.ravel()
AN = D.get("anneal") or {}
BCOL = {0: C["kov"], 200: C["teal"], 1000: C["cold"]}

# (a) anneal trajectories, with the free excursion (grey) and, dashed, the gap that a clock common to both networks would give:
# the free excursion read at the intrinsic time B + s(u), s(u) = u - u^2/1000 after u steps of the 500-step linear anneal;
# dots mark the end of each anneal (filled: observed; open: shared-clock prediction)
FR = D.get("free")
if FR: a.plot(FR["t"], 1e3 * np.array(FR["d"]), color=C["grey"], lw=0.9, label="no anneal")
ANs = sorted(AN.items(), key=lambda kv: int(kv[0][1:]))
for key, v in ANs:
    B = int(key[1:]); cu = np.array(v["eval_loss"]["curve"], float); col = BCOL.get(B, C["grey"])
    a.plot(cu[:, 0], 1e3 * cu[:, 1], color=col, lw=1.2, label=rf"anneal, $t_\mathrm{{d}}$ = {B}")
    a.plot(cu[-1, 0], 1e3 * cu[-1, 1], "o", ms=3.5, color=col, zorder=5)
    if FR:
        u = np.clip(cu[:, 0] - B, 0, 500); s_int = np.where(cu[:, 0] > B, B + u - u * (u - 1) / 1000, cu[:, 0])
        pr = 1e3 * np.interp(s_int, FR["t"], FR["d"])
        a.plot(cu[:, 0], pr, color=col, lw=0.8, ls="--")
        a.plot(cu[-1, 0], pr[-1], "o", ms=3.5, mfc="white", mec=col, mew=0.9, zorder=5)
a.axhline(0, color=C["grey"], lw=0.6); a.set_xlabel("steps after match"); a.set_ylabel(r"$\Delta L$ ($10^{-3}$ nats)")
if AN:
    from matplotlib.lines import Line2D
    h_, l_ = a.get_legend_handles_labels()
    h_.append(Line2D([], [], color=C["ref"], lw=0.8, ls="--")); l_.append("shared clock")
    lo_, hi_ = a.get_ylim(); a.set_ylim(lo_, lo_ + (hi_ - lo_) / 0.55); a.legend(h_, l_, loc="upper right", ncol=1)
label(a, "a")

# (b) final differences with block-bootstrap 95% intervals (filled: set A; open: set B), coloured by B as in (a); the bar
# marks the shared-clock prediction, the free excursion at intrinsic time B + 250.5
for i, (key, v) in enumerate(ANs):
    B = int(key[1:]); col = BCOL.get(B, C["grey"])
    for j, (S, mk, fill) in enumerate((("A", "o", col), ("B", "s", "white"))):
        if f"{S}_last_diff" not in v: continue
        y = 1e3 * v[f"{S}_last_diff"]; ci = 1e3 * np.array(v[f"{S}_last_ci"])
        b.errorbar([i + (j - 0.5) * 0.24], [y], yerr=[[y - ci[0]], [ci[1] - y]], fmt=mk, ms=4.5, color=col, mfc=fill,
                   mec=col, mew=0.9, capsize=2, lw=1.0)
        if FR:
            fd = np.array(FR["d"] if S == "A" else FR.get("d_b", FR["d"]))
            b.plot([i + (j - 0.5) * 0.24 - 0.1, i + (j - 0.5) * 0.24 + 0.1], [1e3 * np.interp(B + 250.5, FR["t"], fd)] * 2,
                   color=col, lw=1.4)
# the Kovacs checkpoint after the function-preserving norm reductions, annealed from the match against the unchanged
# reference (k_annealN.py, G_N on set A): rescaling leaves the annealed gap unchanged
GN = (D.get("annealN") or {}).get("G_N")
if GN and "A_last_diff" in GN:
    y = 1e3 * GN["A_last_diff"]; ci = 1e3 * np.array(GN["A_last_ci"])
    b.errorbar([0.36], [y], yerr=[[y - ci[0]], [ci[1] - y]], fmt="D", ms=4.0, color=C["gold"], mfc=C["gold"], mec=C["gold"],
               capsize=2, lw=1.0, zorder=5)
b.axhline(0, color=C["grey"], lw=0.6); b.set_xticks(range(len(ANs))); b.set_xticklabels([k[1:] for k, _ in ANs])
b.set_xlim(-0.6, max(len(ANs), 2) - 0.4)
b.set_xlabel(r"steps before anneal, $t_\mathrm{d}$"); b.set_ylabel(r"final $\Delta L$ ($10^{-3}$ nats)")
if AN:
    from matplotlib.lines import Line2D
    hb = [Line2D([], [], ls="", marker="o", ms=4.5, color=C["ref"]), Line2D([], [], ls="", marker="s", ms=4.5, mfc="white",
          mec=C["ref"], mew=0.9), Line2D([], [], color=C["ref"], lw=1.4)]
    lb = ["set A", "set B", "shared clock"]
    if GN: hb.append(Line2D([], [], ls="", marker="D", ms=4.0, color=C["gold"])); lb.append("rescaled Kovacs (A)")
    lo_, hi_ = b.get_ylim(); b.set_ylim(min(lo_, 0), hi_ / 0.66)
    b.legend(hb, lb, loc="upper right", ncol=2, columnspacing=0.8, handletextpad=0.4)
label(b, "b")

# (c) independent matched pairs, each annealed from its match (t_d = 0): final difference with joint block-bootstrap 95%
# intervals (filled: set A; open: set B), the shared-clock prediction of the same pair (bar) and its largest free
# excursion (grey diamond)
P = D.get("pairs") or {}
for i, (nm, e) in enumerate(P.items()):
    for j, (S, mk, fill) in enumerate((("A", "o", C["kov"]), ("B", "s", "white"))):
        if f"{S}_final_last" not in e: continue
        x = i + (j - 0.5) * 0.26; y = 1e3 * e[f"{S}_final_last"]; ci = 1e3 * np.array(e[f"{S}_final_last_ci"])
        c.errorbar([x], [y], yerr=[[y - ci[0]], [ci[1] - y]], fmt=mk, ms=4.5, color=C["kov"], mfc=fill, mec=C["kov"],
                   mew=0.9, capsize=2, lw=1.0, zorder=4)
        c.plot([x - 0.1, x + 0.1], [1e3 * e[f"{S}_clock"]] * 2, color=C["kov"], lw=1.4)
        c.plot([x], [1e3 * e[f"{S}_fmax"]], ls="", marker="D", ms=3.2, color=C["grey"], zorder=3)
c.axhline(0, color=C["grey"], lw=0.6); c.set_xticks(range(len(P)))
c.set_xticklabels(list(P), fontsize=0.85 * plt.rcParams["xtick.labelsize"] if len(P) > 3 else None)   # four pairs: keep the labels apart
c.set_xlim(-0.6, max(len(P), 2) - 0.4); c.set_xlabel("matched pair"); c.set_ylabel(r"final $\Delta L$ ($10^{-3}$ nats)")
if P:
    from matplotlib.lines import Line2D
    hc = [Line2D([], [], ls="", marker="o", ms=4.5, color=C["kov"]), Line2D([], [], ls="", marker="s", ms=4.5,
          mfc="white", mec=C["kov"], mew=0.9), Line2D([], [], color=C["kov"], lw=1.4),
          Line2D([], [], ls="", marker="D", ms=3.2, color=C["grey"])]
    lo_, hi_ = c.get_ylim(); c.set_ylim(min(lo_, 0), hi_ / 0.72)
    c.legend(hc, ["set A", "set B", "shared clock", "free maximum"], loc="upper right", ncol=2)
label(c, "c")

# (d) ageing
for k, nm, col in (("young", "from step 4000", C["kov"]), ("old", "from step 8000", C["gold"])):
    g = D.get("age", {}).get(k)
    if g:
        t, y = logbin(g["t"], g["d"]); d.plot(t, 1e3 * y, color=col, lw=1.2, label=nm)
logx(d); d.set_xlabel("steps after match"); d.set_ylabel(r"$\Delta L$ ($10^{-3}$ nats)")
if d.get_legend_handles_labels()[0]:
    lo_, hi_ = d.get_ylim(); d.set_ylim(lo_, lo_ + (hi_ - lo_) / 0.6); d.legend(loc="upper right")
label(d, "d")

save(fig, stem)
print("saved", stem)
