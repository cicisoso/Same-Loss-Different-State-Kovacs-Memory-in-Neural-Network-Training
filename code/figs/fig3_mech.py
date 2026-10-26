# State-interventions figure (Figure 2 of the manuscript). (a) Optimizer-state transplant at the match: Kovacs weights with
# their own or the reference's Adam state, and reference weights with the Kovacs state, against the reference weights with
# their own state; (b) additive decomposition of the area over the first 300 steps into the simple weight effect (Kovacs
# weights in the reference state), the simple state effect (Kovacs state on the reference weights) and their interaction,
# which add up to the observed excursion (averaged main effects would share the interaction between them and are not shown);
# (c) norms of the weight-matrix groups of the Kovacs network relative to the reference through heating, cooling and release;
# (d) the Kovacs network after function-preserving norm reductions at the match; (e) the reference after the converse norm
# increases; (f) angular update per step of each matrix group, Kovacs relative to reference, without and with the reduction.
# Usage: python fig3_mech.py <fig3_data.json> <out_stem>
import sys, json, numpy as np
from figstyle import plt, C, W2, label, save
from matplotlib.ticker import FixedLocator, FuncFormatter

D = json.load(open(sys.argv[1])); stem = sys.argv[2]
logfmt = FuncFormatter(lambda v, _: f"{v:g}")


def logbin(t, y, per_decade=12):
    """display only: means over logarithmically spaced bins of steps after the match (statistics use the raw evaluations)"""
    t, y = np.asarray(t, float), np.asarray(y, float); m = t >= 1; t, y = t[m], y[m]
    edges = 10 ** np.arange(0, np.log10(t.max()) + 1 / per_decade, 1 / per_decade); k = np.digitize(t, edges)
    tb = np.array([np.exp(np.log(t[k == i]).mean()) for i in np.unique(k)]); yb = np.array([y[k == i].mean() for i in np.unique(k)])
    return tb, yb


def headroom(ax, frac):
    """raise the upper y limit so the data fill the lower `frac` of the axes, leaving the top for the legend"""
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, lo + (hi - lo) / frac)


def logx(ax, hi=1500):
    ax.set_xscale("log"); ax.set_xlim(1, hi); ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(logfmt); ax.axhline(0, color=C["grey"], lw=0.6)


fig, axs = plt.subplots(2, 3, figsize=(W2, 5.3), gridspec_kw={"wspace": 0.62, "hspace": 0.55})
a, b, c, d, e, f = axs.ravel()

# (a) transplant
S = D.get("swap")
if S and "eval_loss" in S:
    t = np.array(S["eval_loss"]["t"]); cu = S["eval_loss"]["curves"]
    for k, nm, col, ls in (("own", "K / K", C["kov"], "-"), ("weights", "K / R", C["hot"], "--"),
                           ("state", "R / K", C["teal"], "-")):
        a.plot(*logbin(t, cu[k]), color=col, ls=ls, lw=1.3 if k == "own" else 1.1, label=nm)
logx(a); a.set_xlabel("steps after match"); a.set_ylabel(r"$\Delta L$ (nats)")
if S and "eval_loss" in S: headroom(a, 0.55); a.legend(loc="upper right", title="weights / state")
label(a, "a")

# (b) additive decomposition of the area (0-300 steps): own = weights + state + interaction (simple effects, Y(R,R) = 0)
if S and "eval_loss" in S:
    st_ = S["eval_loss"]["stats"]
    items = (("own", "total", C["kov"]), ("weights", "weights", C["hot"]), ("state", "state", C["teal"]),
             ("I", "interaction", C["grey"]))
    vals = [st_[k]["area"] for k, _, _ in items]
    if all(v is not None for v in vals):                       # horizontal bars: room for the labels
        y = np.arange(len(items))[::-1]
        b.barh(y, vals, height=0.6, color=[col for _, _, col in items], lw=0)
        b.set_yticks(y); b.set_yticklabels([nm for _, nm, _ in items]); b.set_ylim(-0.6, len(items) - 0.4)
        b.axhline(y[0] - 0.5, color=C["grey"], lw=0.6, ls=":")
b.axvline(0, color=C["grey"], lw=0.6); b.set_xlabel("area (nats\u00b7steps)")
label(b, "b")

# (c) norm ratios through the protocol
N = D.get("norms")
if N:
    st = np.array(N["step"]); cols = [C["gold"], C["hot"], C["hot_soft"], C["cold"], C["cold_soft"]]
    for (g, r), col in zip(N["ratio"].items(), cols):
        c.plot(st, 100 * np.array(r), color=col, lw=1.1, label={"attention in": "attn. in", "attention out": "attn. out"}.get(g, g))
    t0 = N["t0"]                                               # protocol strip along the bottom: heating, cooling
    c.axvspan(0, N["heat_end"] - t0, ymin=0, ymax=0.045, color=C["hot"], lw=0)
    c.axvspan(N["heat_end"] - t0, N["match"] - t0, ymin=0, ymax=0.045, color=C["cold"], lw=0)
    c.set_xlim(0, st.max()); headroom(c, 0.5); c.legend(loc="upper right", ncol=1)
c.axhline(0, color=C["grey"], lw=0.6); c.set_xlabel("training steps"); c.set_ylabel("norm, Kovacs vs ref. (%)")
label(c, "c")

# (d) Kovacs network after norm reductions, (e) reference after norm increases
NM = (D.get("norm") or {}).get("branches", {})
for ax, items in ((d, (("nm_kp_sham", "unchanged", C["kov"], "-"), ("nm_kp_proj", "attn./MLP in", C["hot"], "--"),
                       ("nm_kp_resid", "residual", C["gold"], "--"), ("nm_kp_both", "both", C["green"], "-"))),
                  (e, (("nm_ref_proj", "attn./MLP in", C["hot"], "--"), ("nm_ref_resid", "residual", C["gold"], "--")))):
    for tag, nm, col, ls in items:
        if tag not in NM or "A" not in NM[tag]: continue
        ax.plot(*logbin(NM[tag]["t"], NM[tag]["A"]), color=col, ls=ls, lw=1.2, label=nm)
    logx(ax); ax.set_xlabel("steps after match"); ax.set_ylabel(r"$\Delta L$ (nats)")
    if ax is d: headroom(d, 0.5)
    else:                                                       # room below the (negative) curves for the legend
        lo_, hi_ = ax.get_ylim(); ax.set_ylim(hi_ - (hi_ - lo_) / 0.6, hi_)
    if ax.get_legend_handles_labels()[0]: ax.legend(loc="upper right" if ax is d else "lower left")
label(d, "d"); label(e, "e")

# (f) angular update per step, Kovacs relative to the unchanged reference, early window (1-100 steps after the match)
A = D.get("angle", {})
if "nm_ref_sham" in A:
    ref = A["nm_ref_sham"]; groups = [g for g in ("wte", "wpe", "qkv", "proj", "fc", "out") if g in ref]
    names = {"wte": "token emb.", "wpe": "pos. emb.", "qkv": "attn. in", "proj": "attn. out", "fc": "MLP in", "out": "MLP out"}
    y = np.arange(len(groups))[::-1]; hbar = 0.38
    for i, (tag, nm, col) in enumerate((("nm_kp_sham", "unchanged", C["kov"]), ("nm_kp_both", "norm reduced", C["green"]))):
        if tag not in A: continue
        s0 = np.array(A[tag]["step"]); s0 = s0 - s0.min(); w = s0 <= 100
        sr = np.array(ref["step"]); sr = sr - sr.min(); wr = sr <= 100
        rat = [np.mean(np.array(A[tag][g])[w]) / np.mean(np.array(ref[g])[wr]) for g in groups]
        f.barh(y + (0.5 - i) * hbar, rat, height=hbar, color=col, lw=0, label=nm)
    f.plot([1, 1], [-0.6, len(groups) - 0.5], color=C["grey"], lw=0.6)    # reference level, stopped below the legend
    f.set_yticks(y); f.set_yticklabels([names[g] for g in groups])
    f.set_ylim(-0.6, len(groups) + 2.1); f.set_xlim(0, 1.2); f.legend(loc="upper left", ncol=1)
f.set_xlabel("angular update ratio")
label(f, "f")

save(fig, stem)
print("saved", stem)
