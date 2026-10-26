# E10 analysis: Kovacs humps and their prediction from single-step / single-pulse responses.
# Model (noisy-quadratic "spectral" picture, learning rate as temperature): held-out loss
#   L(t) = R(S1(t)) + sum_j c_j u_j(t),   u_j(t+1) = u_j(t) exp(-eta_t r_j) + eta_t (1 - exp(-eta_t r_j))
# R = "river" progress as a function of the learning-rate area S1 = sum eta; each mode j relaxes at rate eta * r_j towards
# an equilibrium excess proportional to eta. Rates r_j lie on a fixed log grid; weights c_j >= 0 and a river scale are fit
# by NNLS to the single-step and single-pulse runs only, then the Kovacs runs are predicted without refitting.
# Usage: python -I k_an.py <results_dir> <ref_tag> <base_tag> <fit_tags,comma> <test_tags,comma> [out.json]
import sys, json, numpy as np, pandas as pd
from scipy.optimize import nnls, curve_fit

R, ref_tag, base_tag = sys.argv[1], sys.argv[2], sys.argv[3]
fit_tags = [t for t in sys.argv[4].split(",") if t]; test_tags = [t for t in sys.argv[5].split(",") if t]
out_path = sys.argv[6] if len(sys.argv) > 6 else f"{R}/k_an.json"
ld = lambda tag: pd.read_csv(f"{R}/{tag}_eval.tsv", sep="\t")
meta = lambda tag: json.load(open(f"{R}/{tag}_meta.json"))
ref, base = ld(ref_tag), ld(base_tag)
s0 = int(ref.step.min())


def lr_history(tag, until):
    """per-step learning rate from s0 to `until` (exclusive), rebuilt from the run's phases"""
    m = meta(tag); lr = np.zeros(until - s0)
    for ph in m["phases"]:
        a0, a1 = ph["start"] - s0, min(ph["end"], until) - s0
        if a1 > a0: lr[a0:a1] = ph["lr"]
    return lr


# ---- river R(S1): power law fitted on the constant-lr base + reference curve
eta_ref = meta(ref_tag)["phases"][0]["lr"]
wb = meta(base_tag)["phases"][0]["end"]                               # warmup length
cur = pd.concat([base[(base.step >= 2 * wb) & (base.step < s0)], ref]).dropna(subset=["eval_loss"])
S = eta_ref * (cur.step.values - wb / 2.0)
f = lambda S, Linf, A, So, al: Linf + A * (S + So) ** (-al)
pr, _ = curve_fit(f, S, cur.eval_loss.values, p0=[3.0, 2.0, 0.1, 0.5], bounds=([0, 0, 0, 0.01], [10, 1e3, 50, 5]), maxfev=200000)
river = lambda S: f(S, *pr)
S_at_s0 = eta_ref * (s0 - wb / 2.0)

rgrid = np.logspace(np.log10(1 / (20000 * eta_ref)), np.log10(1 / (2 * eta_ref)), 28)        # rates per unit eta*step


PQ = (1.0, 1.0)                                                       # equilibrium excess ~ eta^p, relaxation rate ~ eta^q
def features(lr):
    """modes u_j(t) - 1 (u = excess in units of the eta_ref equilibrium) for an lr history that starts in equilibrium at
    eta_ref; river difference"""
    p_, q_ = PQ; x = lr / eta_ref
    u = np.ones(len(rgrid)); U = np.zeros((len(lr) + 1, len(rgrid))); U[0] = u
    for t, e in enumerate(x):
        q = np.exp(-eta_ref * e ** q_ * rgrid); u = u * q + e ** p_ * (1 - q); U[t + 1] = u
    S1 = S_at_s0 + np.r_[0, np.cumsum(lr)]; S1r = S_at_s0 + eta_ref * np.arange(len(lr) + 1)
    return U - 1.0, river(S1) - river(S1r)


rr = ref.dropna(subset=["eval_loss"]).sort_values("step")
def delta(tag):
    """loss minus the reference, with the reference interpolated linearly between its 5-10-step evals"""
    d = ld(tag).dropna(subset=["eval_loss"]); d = d[(d.step > s0) & (d.step <= rr.step.max())]
    return d.step.values, d.eval_loss.values - np.interp(d.step.values, rr.step.values, rr.eval_loss.values)


info, best = {}, None
fit_data = {tag: delta(tag) for tag in fit_tags}
for pq in [(p_, q_) for p_ in (0.5, 0.75, 1.0, 1.25, 1.5) for q_ in (0.5, 0.75, 1.0, 1.25, 1.5)]:
    PQ = pq; rows, ys = [], []
    for tag in fit_tags:
        st, dl = fit_data[tag]; lr = lr_history(tag, int(st.max()))
        U, rv = features(lr); k = st - s0
        rows.append(np.c_[U[k], rv[k]]); ys.append(dl)
    X = np.vstack(rows); y = np.concatenate(ys)
    coef, _ = nnls(X, y); rmse = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
    if best is None or rmse < best[0]: best = (rmse, pq, coef, len(y))
fit_rmse, PQ, coef, ny = best
c, beta = coef[:-1], coef[-1]
info["fit"] = {"river_params": pr.tolist(), "river_scale": float(beta), "rmse": fit_rmse, "n": int(ny), "p_q": list(PQ),
               "modes": [[float(1 / (r * eta_ref)), float(w)] for r, w in zip(rgrid, c) if w > 0]}  # (tau steps @eta_ref, eq. excess)

for tag in test_tags:
    st, dl = delta(tag); m = meta(tag)
    lr = lr_history(tag, int(st.max())); U, rv = features(lr); k = st - s0
    pred = U[k] @ c + beta * rv[k]
    ph = m["phases"]; rel = next((p["matched_at"] for p in ph if p.get("matched_at")), None)
    r = {"matched_at": rel, "match_gap": next((p.get("match_gap") for p in ph if p.get("matched_at")), None)}
    if rel:
        after = st >= rel; t = st[after] - rel; o = dl[after]; pp = pred[after]
        sgn = 1 if ph[0]["lr"] > eta_ref else -1                       # heat-first protocols should give a positive hump
        io, ip = int(np.argmax(sgn * o)), int(np.argmax(sgn * pp))
        r.update({"hump_obs": float(o[io]), "t_peak_obs": int(t[io]), "hump_pred": float(pp[ip]), "t_peak_pred": int(t[ip]),
                  "end_obs": float(o[-1]), "end_pred": float(pp[-1]),
                  "nrmse": float(np.sqrt(np.mean((o - pp) ** 2)) / max(np.ptp(o), 1e-9)),
                  "curve": [[int(a), float(b), float(cc)] for a, b, cc in zip(t, o, pp)]})
    info[tag] = r
    print(tag, json.dumps({k2: v for k2, v in r.items() if k2 != "curve"}), flush=True)
print("fit", json.dumps({k2: v for k2, v in info["fit"].items() if k2 != "modes"}), flush=True)
print("modes(tau_steps, eq_excess)", [[round(a), round(b, 4)] for a, b in info["fit"]["modes"]], flush=True)
json.dump(info, open(out_path, "w"), indent=1)
