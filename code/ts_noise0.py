# E10 check of the zero-label-noise case of ts_noise.py (Codex round 3): same recursion and preparation (20000 steps at
# eta_ref from c = 1/sum(lam), heat 300 steps, cool until the crossing, release), sigma = 0, B = 16. Reports the crossing,
# the gap at release, and the extremes of the excursion after release.
# Usage: python -I ts_noise0.py
import numpy as np

eta_ref, eta_h, eta_l, B, sig = 0.5, 1.5, 0.0625, 16, 0.0
k = np.arange(1, 4001, dtype=float); lam = k ** -1.5; lam = lam / lam.sum(); c = np.ones_like(lam) / lam.sum(); g = 1.0 / B
step = lambda c, eta: c * (1 - 2 * eta * lam + eta ** 2 * lam ** 2 * (1 + g)) + eta ** 2 * lam * ((lam * c).sum() + sig ** 2) * g
loss = lambda c: 0.5 * (lam * c).sum()
for _ in range(20000): c = step(c, eta_ref)
T = 300 + 40000 + 20000; ref = np.empty(T + 1); cr = c.copy()
for t in range(T + 1): ref[t] = loss(cr); cr = step(cr, eta_ref)
ck = c.copy(); t = 0
for _ in range(300): ck = step(ck, eta_h); t += 1
print("heat excess (mean of last 50 not tracked); loss after heating", loss(ck), "ref", ref[t])
while loss(ck) > ref[t] and t < 300 + 40000: ck = step(ck, eta_l); t += 1
tm = t; dl = []
for _ in range(20000): dl.append(loss(ck) - ref[t]); ck = step(ck, eta_ref); t += 1
dl = np.array(dl)
print("cool steps", tm - 300, "release gap", dl[0], "ref loss at match", ref[tm])
print("max", dl.max(), "at", int(dl.argmax()), "min", dl.min(), "at", int(dl.argmin()), "rel min/ref", dl.min() / ref[tm])
