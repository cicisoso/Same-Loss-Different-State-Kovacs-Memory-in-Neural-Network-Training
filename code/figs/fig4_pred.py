# Forecasts figure: step responses, schedule-conditioned forecasts and the conditional timing law. (a) per-unit step
# responses against intrinsic time; (b) forecasts of the exploratory Kovacs run by schedule-to-loss models fitted to steps and
# pulses only; (c) withheld protocols, observed against the
# spectral model; (d) pre-registered peak-time law, predicted against both registered observed lags; (e) timing errors of
# the law and of the two pilot-calibrated baselines on the withheld runs; (f) registered contest, criterion by criterion.
# Usage: python fig4_pred.py <fig4_data.json> <out_stem>
import sys, json, numpy as np
from figstyle import plt, C, W2, label, save
import matplotlib.gridspec as gs
import matplotlib.colors as mpl_colors
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

D = json.load(open(sys.argv[1])); stem = sys.argv[2]
MOD = [("tissue", "single timescale", C["grey"]), ("mpl", "multi-power", C["gold"]), ("fsl", "functional", C["teal"]),
       ("nqm2", "two modes", C["cold_soft"]), ("nqmS", "26-mode", C["kov"])]
FAMILY = [("kp_ta100", "100 hot"), ("kp_ta1000", "1000 hot"), ("kp_h2", r"hot 2$\eta_{\rm ref}$"),
          ("kp_l2", r"cold $\eta_{\rm ref}/4$")]
FCOL = [C["hot"], C["cold"], C["gold"], C["teal"]]


def smooth(y, k=3):
    y = np.asarray(y, float); p = np.r_[[y[0]] * (k // 2), y, [y[-1]] * (k // 2)]
    return np.convolve(p, np.ones(k) / k, "valid")


def key(run):                                                  # contest keys are "run" or "run:ref"
    for k in D["runs"]:
        if k.split(":")[0] == run: return k
    return None


fig = plt.figure(figsize=(W2, 5.3))
G = gs.GridSpec(2, 3, figure=fig, wspace=0.62, hspace=0.55)
ax = [fig.add_subplot(G[i, j]) for i in range(2) for j in range(3)]
logfmt = FuncFormatter(lambda v, _: f"{v:g}")

# (a) per-unit step responses in intrinsic time
a = ax[0]
for tag, v in sorted(D["steps"].items(), key=lambda kv: kv[1]["eta"]):
    eta = v["eta"]; t = np.asarray(v["t"], float); y = smooth(v["per_unit"]); s = t * eta; m = s > 0
    col = C["hot"] if eta > 1 else C["cold"]; alpha = min(1.0, 0.45 + 0.55 * abs(np.log(eta)) / np.log(8))
    a.plot(s[m], y[m], color=col, lw=0.9, alpha=alpha, ls="-" if eta < 1 else "--",
           label={0.125: "1/8", 0.25: "1/4", 0.5: "1/2"}.get(eta, f"{eta:g}"))
if D.get("logline"):
    ll = D["logline"]; xs = np.array([0.7, 300.0])
    a.plot(xs, ll["c"] + ll["w"] * np.log(xs), color=C["grey"], lw=0.7, ls=":", zorder=0)
a.set_xscale("log"); a.set_xlim(0.5, 3000); a.set_ylim(0, 0.42 if D.get("river_corrected") else 0.3)   # top: legend
a.set_xlabel(r"intrinsic time $s$ (reference steps)"); a.set_ylabel(r"response per unit $\Delta\eta/\eta_{\rm ref}$ (nats)")
a.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000])); a.xaxis.set_major_formatter(logfmt)
a.legend(loc="upper left", ncol=2, title=r"$\eta\,/\,\eta_{\rm ref}$", columnspacing=0.8)
label(a, "a")

# (b) blind predictions of the exploratory Kovacs run
b = ax[1]; kb = key("kp")
if kb:
    for m, nm, col in MOD:
        if m not in D["runs"][kb] or m in ("fsl", "nqm2"): continue          # all five models are scored in (f)
        c = np.asarray(D["runs"][kb][m], float); t = c[:, 0]; w = t >= 1
        if m == MOD[0][0]:
            b.plot(t[w], smooth(c[w, 1]), color="k", lw=1.4, label="observed", zorder=5)
        b.plot(t[w], c[w, 2], color=col, lw=0.9, label=nm, ls="--" if m == "tissue" else "-")
b.axhline(0, color=C["light"], lw=0.5); b.set_xscale("log"); b.set_xlim(1, 2500)
b.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000])); b.xaxis.set_major_formatter(logfmt)
b.set_xlabel("steps after release"); b.set_ylabel(r"$\Delta L$ (nats)")
lo_, hi_ = b.get_ylim(); b.set_ylim(lo_ - 1.0 * (hi_ - lo_), hi_)          # room below the curves for the legend
b.legend(loc="lower right")
label(b, "b")

# (c) withheld protocols: observed (smoothed) and the spectral model's prediction
c_ = ax[2]
for (run, nm), col in zip(FAMILY, FCOL):
    k = key(run)
    if not k or "nqmS" not in D["runs"][k]: continue
    c = np.asarray(D["runs"][k]["nqmS"], float); t = c[:, 0]; w = t >= 1
    c_.plot(t[w], smooth(c[w, 1]), color=col, lw=1.1, label=nm); c_.plot(t[w], c[w, 2], color=col, lw=0.8, ls="--")
c_.axhline(0, color=C["light"], lw=0.5); c_.set_xscale("log"); c_.set_xlim(1, 2500)
c_.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000])); c_.xaxis.set_major_formatter(logfmt)
c_.set_xlabel("steps after release"); c_.set_ylabel(r"$\Delta L$ (nats)")
lo_, hi_ = c_.get_ylim(); c_.set_ylim(lo_ - 1.0 * (hi_ - lo_), hi_)          # room below the curves for the legend
if c_.get_legend_handles_labels()[0]:
    c_.legend(loc="lower left")
label(c_, "c")

# (d) pre-registered peak-time law against both registered observed lags: two-mode-fit t* (filled) and the maximum of the
# smoothed excursion over 10-150 steps (open), joined per run
d = ax[3]; lo, hi = 5, 400
xx = np.geomspace(lo, hi, 60)                                # factor-1.5 band, kept inside the axes
d.fill_between(xx, np.maximum(xx / 1.5, lo), np.minimum(xx * 1.5, hi), color=C["light"], alpha=0.5, lw=0)
d.plot(xx, xx, color=C["grey"], lw=0.6, ls="--")
EXPL = {"kp", "code_kp", "mfn_kp"}
rcol = lambda run: C["kov"] if run.startswith("kp") else (C["teal"] if run.startswith(("code", "pubmed", "big")) else C["gold"])
for run, v in D["peaklaw"].items():
    if v.get("pred") is None: continue
    ex = run in EXPL; col = rcol(run); obs = [o for o in (v.get("t_star"), v.get("t_max")) if o]
    if len(obs) == 2: d.plot([v["pred"]] * 2, obs, color=col, lw=0.6, alpha=0.7)
    if v.get("t_star"): d.plot(v["pred"], v["t_star"], "o", ms=3.4, mfc="white" if ex else col, mec=col, mew=0.8)
    if v.get("t_max"): d.plot(v["pred"], v["t_max"], "s", ms=3.0, mfc="white", mec=col, mew=0.8)
d.set_xscale("log"); d.set_yscale("log"); d.set_xlim(lo, hi); d.set_ylim(lo, hi)
for axis in (d.xaxis, d.yaxis):
    axis.set_major_locator(FixedLocator([10, 30, 100, 300])); axis.set_major_formatter(logfmt); axis.set_minor_locator(NullLocator())
d.set_xlabel(r"predicted lag $\sigma^*$ (steps)"); d.set_ylabel("observed lag (steps)")
label(d, "d")

# (e) timing errors of the law and of the two pilot-calibrated baselines on the withheld runs, against both observed lags
e = ax[4]; PRED = (("pred", "law"), ("base_const", "lag"), ("base_frac", "fraction"))
held = {r: v for r, v in D["peaklaw"].items() if r not in EXPL and v.get("pred") is not None}
e.axhspan(-np.log2(1.5), np.log2(1.5), color=C["light"], alpha=0.5, lw=0); e.axhline(0, color=C["grey"], lw=0.6, ls="--")
rng = np.random.default_rng(1)
for ip, (pk, nm) in enumerate(PRED):
    for io, (ok_, mk) in enumerate((("t_star", "o"), ("t_max", "s"))):
        v = np.array([np.log2(x[pk] / x[ok_]) for x in held.values() if x.get(pk) and x.get(ok_)])
        if not len(v): continue
        xs = ip + (io - 0.5) * 0.3 + rng.uniform(-0.06, 0.06, len(v))
        e.plot(xs, v, mk, ms=3.0, mfc=C["kov"] if io == 0 else "white", mec=C["kov"], mew=0.7, alpha=0.9)
        e.plot([ip + (io - 0.5) * 0.3 - 0.11, ip + (io - 0.5) * 0.3 + 0.11], [np.median(v)] * 2, color="k", lw=1.2)
e.set_xticks(range(len(PRED))); e.set_xticklabels([nm for _, nm in PRED]); e.set_xlim(-0.6, len(PRED) - 0.4)
e.set_ylabel(r"$\log_2$(predicted / observed lag)"); e.tick_params(axis="x", length=0)
e.set_xlabel("timing rule")
label(e, "e")

# (f) registered contest on the withheld Kovacs runs: per frozen implementation, the runs meeting each component of the
# criterion (k_score.py: sign of the maximum; height and lag ratios strictly within a factor 1.5; NRMSE < 0.3) and all four
f = ax[5]; runs = [r for r in D["scores"] if r.split(":")[0] not in EXPL]
MODF = [("tissue", "single"), ("mpl", "multi-power"), ("fsl", "functional"), ("nqm2", "two-mode"),
        ("nqm2pq", r"free $p,q$"), ("nqmS", "26-mode")]
CRIT = ["sign", "height", "lag", "curve", "all"]
K = np.zeros((len(MODF), len(CRIT)), int); N = np.zeros(len(MODF), int)
for i, (m, _) in enumerate(MODF):
    for r in runs:
        v = D["scores"][r].get(m)
        if not v: continue
        N[i] += 1
        K[i] += [np.sign(v["pred"]["peak"]) == np.sign(v["obs"]["peak"]), 1 / 1.5 < v["height_ratio"] < 1.5,
                 1 / 1.5 < v["time_ratio"] < 1.5, v["nrmse"] < 0.3, bool(v["success"])]
frac = K / np.maximum(N, 1)[:, None]
cmap = mpl_colors.LinearSegmentedColormap.from_list("kov", ["white", C["kov"]])
f.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto")
same = len(set(N)) == 1                                           # one denominator: counts in the cells, N in the label
for i in range(len(MODF)):
    for j in range(len(CRIT)):
        f.text(j, i, f"{K[i, j]}" if same else f"{K[i, j]}/{N[i]}", ha="center", va="center",
               color="white" if frac[i, j] > 0.65 else "k", fontweight="bold" if CRIT[j] == "all" else "normal")
f.set_xticks(range(len(CRIT))); f.set_xticklabels(CRIT, rotation=90)
f.set_yticks(range(len(MODF))); f.set_yticklabels([nm for _, nm in MODF])
f.tick_params(length=0); [sp.set_visible(False) for sp in f.spines.values()]
f.set_xlabel(f"criterion met (of {N[0]} runs)" if same else "criterion met (runs)")
label(f, "f")

save(fig, stem)
print("saved", stem, "runs", len(D["runs"]), "peaklaw", len(D["peaklaw"]), "scored", len(runs))
