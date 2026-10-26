# E10 minimal model, spectral separation: online SGD on Gaussian linear regression with two curvature levels, a fast one
# (lambda_1) and a kappa-fold slower one carried by kappa directions so that both hold the same share of the noise-driven
# excess loss. Exact second-moment recursion (see ts_exact.py) started in the stationary state (no drift), Kovacs protocol
# heat (eta_h, ta steps) -> cool (eta_l) until the loss is back at the stationary value -> eta_ref. Compared with the
# linear-response prediction in intrinsic time, dL(s) = sum_j w_j D_j exp(-r_j s), r_j = 2 lambda_j,
# D_j = (eta_h - eta_l) x_j - (eta_h - eta_ref) x_j y_j - (eta_ref - eta_l), x_j = exp(-r_j eta_l t_c), y_j = exp(-r_j eta_h ta)
# (units of the eta_ref equilibrium excess), which is zero for kappa = 1 and positive for every kappa > 1.
# Usage: python -I ts_kappa.py <out.json>
import sys, json, numpy as np

out = sys.argv[1]; B, sig = 16, 0.5
eta_ref, eta_h, eta_l, ta = 0.5, 1.5, 0.0625, 300


def exact(kappa):
    m = np.array([1.0, kappa]); lam = np.array([1.0, 1.0 / kappa]); lam = lam / (m * lam).sum()
    def step(c, eta):
        L2 = (m * lam * c).sum()
        return c * (1 - 2 * eta * lam + eta ** 2 * lam ** 2 * (1 + 1 / B)) + eta ** 2 * lam * (L2 + sig ** 2) / B
    loss = lambda c: 0.5 * (m * lam * c).sum()
    c = np.full(2, eta_ref * sig ** 2 / (2 * B))
    for _ in range(int(60 * kappa / (2 * eta_ref * lam[0])) + 2000): c = step(c, eta_ref)
    L0 = loss(c); heat = []
    for _ in range(ta): c = step(c, eta_h); heat.append(loss(c) - L0)
    tc = 0
    while loss(c) > L0 and tc < 10 ** 6: c = step(c, eta_l); tc += 1
    dl = []
    for _ in range(int(20 * kappa / (2 * eta_ref * lam[0])) + 2000): dl.append(loss(c) - L0); c = step(c, eta_ref)
    dl = np.array(dl); i = int(np.argmax(dl))
    return {"tc": tc, "eq": L0, "heat_excess": heat[-1], "gap": float(dl[0]), "hump": float(dl[i]), "t_peak": i,
            "hump_rel": float(dl[i] / heat[-1]), "lam": lam.tolist()}


def linear(kappa, n=200000):
    """linear response in intrinsic time with the exact model's rates and loss weights; t_c from the linear matching"""
    m = np.array([1.0, kappa]); lam = np.array([1.0, 1.0 / kappa]); lam = lam / (m * lam).sum(); r = 2 * lam
    w = m * lam / (m * lam).sum()                                          # equal shares by construction
    u = np.full(2, eta_ref)                                                # mode excess in units of eta
    for _ in range(ta): u = eta_h + (u - eta_h) * np.exp(-r * eta_h)
    heat = (w * u).sum() - eta_ref; tc = 0
    while (w * u).sum() > eta_ref and tc < n: u = eta_l + (u - eta_l) * np.exp(-r * eta_l); tc += 1
    s = np.arange(0, int(20 * kappa / (2 * eta_ref * lam[0])) + 2000) * eta_ref
    d = ((w * (u - eta_ref))[None, :] * np.exp(-np.outer(s, r))).sum(1); i = int(np.argmax(d))
    return {"tc": tc, "hump_rel": float(d[i] / heat), "t_peak": i}


res = {}
for kappa in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
    e, l = exact(kappa), linear(kappa); res[str(kappa)] = {"exact": e, "linear": l}
    print(kappa, "exact", {k: round(v, 5) if isinstance(v, float) else v for k, v in e.items() if k != "lam"},
          "linear", l, flush=True)
json.dump(res, open(out, "w"), indent=1)
