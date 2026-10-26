# E10 minimal model: online SGD on linear regression with a power-law input spectrum (teacher-student), the exactly solvable
# noisy-quadratic setting. Population loss L = 1/2 sum_k lam_k (w_k - w*_k)^2 + noise/2 is computed exactly from the weights.
# Protocols mirror the GPT runs: base at eta_ref, then reference / Kovacs heat-then-cool / cool-then-heat / single steps.
# Usage: python -I ts_kovacs.py <out.json> [d] [alpha] [batch] [reps]
import sys, json, numpy as np, torch

out = sys.argv[1]; d = int(sys.argv[2]) if len(sys.argv) > 2 else 2000; al = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
B = int(sys.argv[4]) if len(sys.argv) > 4 else 16; reps = int(sys.argv[5]) if len(sys.argv) > 5 else 64
dev = "cuda" if torch.cuda.is_available() else "cpu"
g = torch.Generator(device=dev).manual_seed(0)
lam = torch.arange(1, d + 1, device=dev, dtype=torch.float64) ** (-al); lam = lam / lam.sum()
wstar = torch.arange(1, d + 1, device=dev, dtype=torch.float64) ** (-0.5) * torch.randn(d, device=dev, generator=g, dtype=torch.float64)
sig = 0.5                                                           # label noise sd
eta_ref, eta_h, eta_l = 0.5, 1.5, 0.0625                          # same ratios as the GPT pilot (x3, /8)


def loss(W): return 0.5 * ((W - wstar) ** 2 * lam).sum(1) + 0.5 * sig ** 2


def sgd(W, eta, n, gen, rec=None, every=10, t0=0):
    sl = lam.sqrt()
    for t in range(n):
        X = torch.randn(W.shape[0], B, d, device=dev, generator=gen, dtype=torch.float64) * sl
        y = (X * wstar).sum(2) + sig * torch.randn(W.shape[0], B, device=dev, generator=gen, dtype=torch.float64)
        r = (X * W[:, None, :]).sum(2) - y
        W = W - eta * (X * r[:, :, None]).mean(1)
        if rec is not None and (t + 1) % every == 0: rec.append((t0 + t + 1, loss(W).mean().item()))
    return W


W0 = torch.zeros(reps, d, device=dev, dtype=torch.float64)
gb = torch.Generator(device=dev).manual_seed(1)
W0 = sgd(W0, eta_ref, 20000, gb)                                    # base: long training at eta_ref
res = {"d": d, "alpha": al, "B": B, "reps": reps, "eta": [eta_ref, eta_h, eta_l]}
seed = 7
def run(phases, match=None, n_after=6000, tag=""):
    gen = torch.Generator(device=dev).manual_seed(seed); W = W0.clone(); rec = [(0, loss(W).mean().item())]; t = 0; info = {}
    for eta, n in phases:
        W = sgd(W, eta, n, gen, rec, 10, t); t += n
    if match is not None:
        eta, refcurve = match; sign0 = np.sign(rec[-1][1] - refcurve[t]); k = 0
        while k < 40000:
            W = sgd(W, eta, 10, gen, rec, 10, t); t += 10; k += 10
            if np.sign(rec[-1][1] - refcurve[t]) != sign0: break
        info["matched_at"] = t; info["gap"] = rec[-1][1] - refcurve[t]
    W = sgd(W, eta_ref, n_after, gen, rec, 10, t)
    info["curve"] = rec; return info

ref = run([], None, 50000)
refc = {s: v for s, v in ref["curve"]}
res["ref"] = ref
res["kp"] = run([(eta_h, 300)], (eta_l, refc), 6000)
res["km"] = run([(eta_l, 300)], (eta_h, refc), 6000)
for e in (eta_h, eta_l, 2 * eta_ref, eta_ref / 2):
    res[f"step_{e:g}"] = run([(e, 6000)], None, 0)
for k in ("kp", "km"):
    m = res[k]["matched_at"]; dl = [(s - m, v - refc[s]) for s, v in res[k]["curve"] if s >= m and s in refc]
    i = int(np.argmax([abs(v) for _, v in dl])); print(k, "matched_at", m, "gap", res[k]["gap"], "peak", dl[i], flush=True)
json.dump(res, open(out, "w"))
