# E10 minimal model, spectral interventions (Codex eval #6): exact second-moment dynamics of online SGD on Gaussian linear
# regression (see ts_exact.py). The heat->cool->release Kovacs protocol is run on spectra of different width and with the fast
# or slow sector removed; a hump needs modes on both sides of the matched state, so it should vanish for a single mode and
# shrink when either sector is cut. Prints hump height (relative to the heating excess) and peak time.
# Usage: python -I ts_sectors.py <out.json>
import sys, json, numpy as np

out = sys.argv[1]; B, sig = 16, 0.5
eta_ref, eta_h, eta_l = 0.5, 1.5, 0.0625


def run(lam, c_init, ta=300, n_after=20000, base=20000):
    lam = lam / lam.sum(); c = c_init / (lam * c_init).sum()
    def step(c, eta):
        L2 = (lam * c).sum()
        return c * (1 - 2 * eta * lam + eta ** 2 * lam ** 2 * (1 + 1 / B)) + eta ** 2 * lam * (L2 + sig ** 2) / B
    loss = lambda c: 0.5 * (lam * c).sum()
    for _ in range(base): c = step(c, eta_ref)
    T = ta + 40000 + n_after; ref = np.empty(T + 1); cr = c.copy()
    for t in range(T + 1): ref[t] = loss(cr); cr = step(cr, eta_ref)
    ck = c.copy(); t = 0; heat = []
    for _ in range(ta): ck = step(ck, eta_h); t += 1; heat.append(loss(ck) - ref[t])
    while loss(ck) > ref[t] and t < ta + 40000: ck = step(ck, eta_l); t += 1
    tm = t; dl = []
    for _ in range(n_after): dl.append(loss(ck) - ref[t]); ck = step(ck, eta_ref); t += 1
    dl = np.array(dl); i = int(np.argmax(dl))
    return {"cool_steps": tm - ta, "gap": float(dl[0]), "heat_excess": float(np.mean(heat[-50:])), "hump": float(dl[i]),
            "hump_rel": float(dl[i] / max(np.mean(heat[-50:]), 1e-12)), "t_peak": i,
            "dl": dl[[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 19999]].tolist()}


res = {}
k = np.arange(1, 4001, dtype=float)
for al in (0.0, 0.5, 1.0, 1.5, 2.0):                                 # width of the curvature spectrum: lam_k ~ k^-alpha
    res[f"alpha_{al}"] = run(k ** -al, np.ones_like(k)); print("alpha", al, {q: res[f"alpha_{al}"][q] for q in ("cool_steps", "hump_rel", "t_peak")}, flush=True)
lam = k ** -1.5
for nm, sel in (("full", k > 0), ("no_fast", k > 20), ("no_slow", k <= 200), ("two_modes", np.isin(k, [1, 400]))):
    r = run(lam[sel], np.ones(sel.sum())); res[nm] = r; print(nm, {q: r[q] for q in ("cool_steps", "hump_rel", "t_peak")}, flush=True)
json.dump(res, open(out, "w"), indent=1)
