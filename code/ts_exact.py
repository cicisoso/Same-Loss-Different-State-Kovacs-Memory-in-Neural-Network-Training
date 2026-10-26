# E10 minimal model, exact expectations: online SGD (batch B) on Gaussian linear regression with a power-law spectrum.
# For c_k = E[(w_k - w*_k)^2] the diagonal second-moment recursion is closed (Gaussian inputs):
#   c_k <- c_k (1 - 2 eta lam_k + eta^2 lam_k^2 (1 + 1/B)) + eta^2 lam_k (sum_j lam_j c_j + sigma^2) / B   [sum includes j = k]
# Loss L = 1/2 sum_k lam_k c_k + sigma^2/2. No sampling noise, so Kovacs humps can be resolved to any precision.
# Protocol grid: heat (x3) for t_a in {100, 300, 1000, 3000} then cool (/8) until matched, release at eta_ref; and the reverse.
# Usage: python -I ts_exact.py <out.json> [alpha_spectrum] [beta_teacher]
import sys, json, numpy as np

out = sys.argv[1]; al = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5; be = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
d, B, sig = 20000, 16, 0.5
k = np.arange(1, d + 1); lam = k ** -al; lam /= lam.sum()
c0 = k ** -be; c0 = c0 / (lam * c0).sum()                       # initial error per mode (teacher power spectrum)
eta_ref, eta_h, eta_l = 0.5, 1.5, 0.0625


def step(c, eta, n=1):
    for _ in range(n):
        L2 = (lam * c).sum()
        c = c * (1 - 2 * eta * lam + eta ** 2 * lam ** 2 * (1 + 1 / B)) + eta ** 2 * lam * (L2 + sig ** 2) / B
    return c

loss = lambda c: 0.5 * (lam * c).sum() + 0.5 * sig ** 2
c = step(c0.copy(), eta_ref, 20000)                              # base
T = 30000
ref = np.zeros(T + 1); cr = c.copy(); ref[0] = loss(cr)
for t in range(T): cr = step(cr, eta_ref); ref[t + 1] = loss(cr)
res = {"alpha": al, "beta": be, "B": B, "ref": ref[::10].tolist()}


def kovacs(first, second, ta, n_after=20000):
    cc = c.copy(); t = 0
    for _ in range(ta): cc = step(cc, first); t += 1
    s0 = np.sign(loss(cc) - ref[t])
    while t < T - n_after and np.sign(loss(cc) - ref[t]) == s0: cc = step(cc, second); t += 1
    tm = t; dl = []
    for j in range(n_after):
        dl.append(loss(cc) - ref[t]); cc = step(cc, eta_ref); t += 1
    dl = np.array(dl); i = int(np.argmax(np.abs(dl[1:]))) + 1
    return {"t_a": ta, "matched_at": tm, "gap0": float(dl[0]), "t_peak": i, "peak": float(dl[i]), "dl": dl[:20000:5].tolist()}

for ta in (100, 300, 1000, 3000):
    for nm, f, s in (("kp", eta_h, eta_l), ("km", eta_l, eta_h)):
        r = kovacs(f, s, ta); res[f"{nm}_{ta}"] = r
        print(nm, ta, {q: r[q] for q in ("matched_at", "gap0", "t_peak", "peak")}, "excess_scale", loss(step(c.copy(), eta_h, 3000)) - ref[3000], flush=True)
json.dump(res, open(out, "w"))
