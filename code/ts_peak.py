# E10 minimal model, Kovacs peak-time law. For a power-law density of relaxation rates, excess weight a(r) ~ r^(nu-1), linear
# relaxation in intrinsic time gives dE(sigma) = (g_h - g_l) A(s_c + sigma) - (g_h - g_0) A(s_c + s_a + sigma) - (g_0 - g_l) A(sigma)
# with A(x) ~ x^-nu (log x for nu = 0), so the peak lag sigma* depends only on the protocol levels, the intrinsic heating and
# cooling durations s_a, s_c and nu; for s_a -> inf, sigma* = s_c / (Q^(1/(1+nu)) - 1), Q = (g_h - g_l)/(g_0 - g_l).
# Exact SGD second-moment recursion on lambda_k = k^-alpha (nu = 1 - 1/alpha), protocols varying heat level, cool level and
# heating time; compares the observed peak step with the law (g = eta, intrinsic time = steps x eta / eta_ref).
# Usage: python -I ts_peak.py <out.json>
import sys, json, numpy as np
from scipy.optimize import brentq

out = sys.argv[1]; B, sig, eta0 = 16, 0.5, 0.5


def sigma_star(gh, g0, gl, sa, sc, nu):
    A = (lambda x: -np.log(x)) if abs(nu) < 1e-9 else (lambda x: x ** (-nu) / nu)
    dA = lambda x: -x ** (-1 - nu)
    f = lambda s: (gh - gl) * dA(sc + s) - (gh - g0) * dA(sc + sa + s) - (g0 - gl) * dA(s)   # d(dE)/d(sigma)
    return brentq(lambda s: f(s), 1e-6 * sc, 1e3 * (sc + sa))


def run(alpha, eh, el, ta, n_after=6000, base=60000, cmax=60000):
    k = np.arange(1, 4001, dtype=float); lam = k ** -alpha; lam = lam / lam.sum()
    def step(c, eta):
        return c * (1 - 2 * eta * lam + eta ** 2 * lam ** 2 * (1 + 1 / B)) + eta ** 2 * lam * ((lam * c).sum() + sig ** 2) / B
    loss = lambda c: 0.5 * (lam * c).sum()
    c = np.ones_like(lam) / lam.sum()
    for _ in range(base): c = step(c, eta0)
    T = ta + cmax + n_after + 1; ref = np.empty(T); cr = c.copy()
    for t in range(T): ref[t] = loss(cr); cr = step(cr, eta0)
    ck = c.copy(); t = 0
    for _ in range(ta): ck = step(ck, eh); t += 1
    if loss(ck) <= ref[t]: return None
    while loss(ck) > ref[t] and t < ta + cmax: ck = step(ck, el); t += 1
    if t >= ta + cmax: return None
    tc = t - ta; dl = []
    for _ in range(n_after): dl.append(loss(ck) - ref[t]); ck = step(ck, eta0); t += 1
    dl = np.array(dl); i = int(np.argmax(dl))
    return {"tc": tc, "t_peak": i, "hump": float(dl[i])}


res = []
for alpha in (1.25, 1.5, 2.0, 3.0):
    nu = 1 - 1 / alpha
    for eh, el, ta in ((1.5, 0.0625, 300), (1.5, 0.0625, 100), (1.5, 0.0625, 1000), (1.0, 0.0625, 300), (1.5, 0.125, 300),
                       (0.75, 0.0625, 300), (1.5, 0.03125, 300)):
        r = run(alpha, eh, el, ta)
        if r is None: print(alpha, eh, el, ta, "no match", flush=True); continue
        gh, g0, gl = eh / eta0, 1.0, el / eta0
        sa, sc = ta * eh / eta0, r["tc"] * el / eta0
        pred = sigma_star(gh, g0, gl, sa, sc, nu); pred0 = sigma_star(gh, g0, gl, sa, sc, 0.0)
        r.update({"alpha": alpha, "nu": nu, "eta_h": eh, "eta_l": el, "ta": ta, "s_a": sa, "s_c": sc, "pred": pred,
                  "pred_logu": pred0, "pred_inf": sc / (((gh - gl) / (g0 - gl)) ** (1 / (1 + nu)) - 1)})
        res.append(r); print(json.dumps({q: (round(v, 3) if isinstance(v, float) else v) for q, v in r.items()}), flush=True)
json.dump(res, open(out, "w"), indent=1)
