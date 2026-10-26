# Figure S1: heating the convolutional network is a catapult. (a) Held-out loss of the CIFAR-10 Kovacs run and its reference
# through heating (3 eta_ref), cooling (eta_ref/8) and release, logarithmic scale; (b) the weight norm of the Kovacs run
# relative to the reference over the same steps; (c) the transformer pilot for comparison (held-out loss difference);
# (d) the filter-norm exchange: continuations of the second matched CNN pair on identical minibatches, with their own or
# the other network's filter norms, against the reference's own continuation (as fig5_data.py; summarised in Figure 5f).
# Usage: python figS1_cifar.py <results_dir> <out_stem>
import sys, json, numpy as np, pandas as pd
from figstyle import plt, C, W2, label, save

R, stem = sys.argv[1], sys.argv[2]
ld = lambda t: pd.read_csv(f"{R}/{t}_eval.tsv", sep="\t").dropna(subset=["eval_loss"]).sort_values("step")
fig, axs = plt.subplots(2, 2, figsize=(W2, 4.9), gridspec_kw={"wspace": 0.42, "hspace": 0.55})
a, b, c, d = axs.ravel()


def shade(ax, ph, t0):
    ax.axvspan(ph[0]["start"] - t0, ph[0]["end"] - t0, color=C["hot_soft"], alpha=0.3, lw=0)
    ax.axvspan(ph[1]["start"] - t0, ph[1]["end"] - t0, color=C["cold_soft"], alpha=0.3, lw=0)


k, r, m = ld("cifar_kp"), ld("cifar_refL"), json.load(open(f"{R}/cifar_kp_meta.json")); ph = m["phases"]; t0 = ph[0]["start"]
k, r = k[k.step - t0 <= 1200], r[(r.step >= t0) & (r.step - t0 <= 1200)]       # only the plotted window
a.plot(r.step - t0, r.eval_loss, color=C["ref"], lw=1.0, label="reference")
a.plot(k.step - t0, k.eval_loss, color=C["kov"], lw=1.2, label="Kovacs")
a.set_yscale("log"); shade(a, ph, t0); a.set_xlim(0, 1200); a.set_xlabel("steps after checkpoint"); a.set_ylabel("held-out loss (nats)")
a.legend(loc="upper right"); label(a, "a")

rw = ld("cifar_refL")                                    # full reference for the interpolation of its weight norm
w = 100 * (k.wnorm.values / np.interp(k.step, rw.step, rw.wnorm) - 1)
b.plot(k.step - t0, w, color=C["kov"], lw=1.2); shade(b, ph, t0); b.axhline(0, color=C["grey"], lw=0.6)
b.set_xlim(0, 1200); b.set_xlabel("steps after checkpoint"); b.set_ylabel("weight norm vs ref. (%)"); label(b, "b")

k2, r2, m2 = ld("kp"), ld("ref"), json.load(open(f"{R}/kp_meta.json")); ph2 = m2["phases"]; t02 = ph2[0]["start"]
k2 = k2[(k2.step - t02 <= 2000) & (k2.step <= r2.step.max())]
d2 = k2.eval_loss.values - np.interp(k2.step, r2.step, r2.eval_loss)
c.plot(k2.step - t02, d2, color=C["kov"], lw=1.2); shade(c, ph2, t02); c.axhline(0, color=C["grey"], lw=0.6)
c.set_xlim(0, 2000); c.set_xlabel("steps after checkpoint"); c.set_ylabel(r"$\Delta L$, transformer (nats)"); label(c, "c")

# (d) filter-norm exchange of the CNN (display: means over logarithmic bins of time)
def logbin(t, y, per_decade=6):
    t, y = np.asarray(t, float), np.asarray(y, float); m = t >= 1; t, y = t[m], y[m]
    edges = 10 ** np.arange(0, np.log10(t.max()) + 1 / per_decade, 1 / per_decade); kk = np.digitize(t, edges)
    return (np.array([np.exp(np.log(t[kk == i]).mean()) for i in np.unique(kk)]), np.array([y[kk == i].mean() for i in np.unique(kk)]))


try:
    bref = ld("cifar_cn_ref"); tb0 = int(bref.step.min())
    for tag, nm, col, ls in (("cifar_cn_own", "Kovacs", C["hot"], "-"), ("cifar_cn_norm", "Kovacs, ref. norms", C["kov"], "--"),
                             ("cifar_cn_refnorm", "ref., Kovacs norms", C["cold"], "--")):
        q = ld(tag); q = q[q.step <= bref.step.max()]
        d.plot(*logbin(q.step.values - tb0, q.eval_loss.values - np.interp(q.step, bref.step, bref.eval_loss)), color=col, ls=ls,
               lw=1.2, label=nm)
    from matplotlib.ticker import FixedLocator, FuncFormatter                  # plain tick labels, as in the main figures
    d.set_xscale("log"); d.set_xlim(1, 2500); d.axhline(0, color=C["grey"], lw=0.6)
    d.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000])); d.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    lo, hi = d.get_ylim(); d.set_ylim(lo, lo + (hi - lo) / 0.6); d.legend(loc="upper right")
except FileNotFoundError:
    pass
d.set_xlabel("steps after match"); d.set_ylabel(r"$\Delta L$ vs ref. (nats)"); label(d, "d")

save(fig, stem)
print("saved", stem)
