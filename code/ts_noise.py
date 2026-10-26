# E10 minimal model, noise control (Codex eval #5): exact second-moment dynamics of online SGD on Gaussian linear regression
# (see ts_exact.py), heat->cool->release Kovacs protocol at different batch sizes B and label-noise levels sigma with the
# drift (the noise-free gradient flow) unchanged. The excess loss that the protocol manipulates is gradient noise; the hump
# should scale with the noise strength (~1/B) at fixed shape and vanish for full-batch gradient descent (B = inf).
# Usage: python -I ts_noise.py <out.json>
import sys, json, numpy as np

out = sys.argv[1]
eta_ref, eta_h, eta_l = 0.5, 1.5, 0.0625


def run(lam, B, sig, ta=300, n_after=20000, base=20000, cmax=40000):
    lam = lam / lam.sum(); c = np.ones_like(lam) / lam.sum(); g = 0.0 if B == np.inf else 1.0 / B
    def step(c, eta):
        L2 = (lam * c).sum()
        return c * (1 - 2 * eta * lam + eta ** 2 * lam ** 2 * (1 + g)) + eta ** 2 * lam * (L2 + sig ** 2) * g
    loss = lambda c: 0.5 * (lam * c).sum()
    for _ in range(base): c = step(c, eta_ref)
    T = ta + cmax + n_after; ref = np.empty(T + 1); cr = c.copy()
    for t in range(T + 1): ref[t] = loss(cr); cr = step(cr, eta_ref)
    ck = c.copy(); t = 0; heat = []
    for _ in range(ta): ck = step(ck, eta_h); t += 1; heat.append(loss(ck) - ref[t])
    hx = float(np.mean(heat[-50:]))
    if hx <= 0: return {"heat_excess": hx, "note": "heating lowers the loss: no Kovacs protocol"}
    while loss(ck) > ref[t] and t < ta + cmax: ck = step(ck, eta_l); t += 1
    tm = t; dl = []
    for _ in range(n_after): dl.append(loss(ck) - ref[t]); ck = step(ck, eta_ref); t += 1
    dl = np.array(dl); i = int(np.argmax(dl))
    return {"cool_steps": tm - ta, "matched": bool(tm < ta + cmax), "heat_excess": hx, "hump": float(dl[i]),
            "hump_rel": float(dl[i] / hx), "t_peak": i, "ref_loss": float(ref[tm])}


k = np.arange(1, 4001, dtype=float); lam = k ** -1.5; res = {}
for B in (4, 16, 64, 256, np.inf):
    r = run(lam, B, 0.5); res[f"B_{B}"] = r; print("B", B, r, flush=True)
for sig in (0.0, 0.25, 1.0):
    r = run(lam, 16, sig); res[f"sig_{sig}"] = r; print("sigma", sig, r, flush=True)
json.dump(res, open(out, "w"), indent=1)
