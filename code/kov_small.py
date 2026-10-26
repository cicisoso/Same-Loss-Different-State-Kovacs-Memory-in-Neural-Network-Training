# E10 universality: the Kovacs learning-rate protocols of kov_train.py applied to two other learning machines.
#   cifar  ResNet-style CNN on CIFAR-10 (HF uoft-cs/cifar10 parquet), random-crop/flip augmentation drawn per step on GPU
#   mf     matrix-factorisation recommender on MovieLens-1M ratings (biased MF, squared loss, L2), held-out 10% of ratings
# Same schedule JSON and match logic as kov_train.py; every run from the same checkpoint sees the same minibatches.
# Usage: python -I kov_small.py --task cifar|mf --out <tag> --sched '<json>' [--init ckpt] [--save-at s] [--ref ref_eval.tsv]
import argparse, json, os, time, io
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--task", required=True); p.add_argument("--out", required=True); p.add_argument("--sched", required=True)
p.add_argument("--init", default=""); p.add_argument("--save-at", default=""); p.add_argument("--ref", default="")
p.add_argument("--data", default="/home/soso/advsci/data/kov"); p.add_argument("--res", default="/home/soso/advsci/results/kov")
p.add_argument("--ckdir", default="/home/soso/advsci/ckpt/kov"); p.add_argument("--opt", default="sgd")
p.add_argument("--bs", type=int, default=256); p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--mom", type=float, default=0.9); p.add_argument("--dim", type=int, default=64)
p.add_argument("--width", type=int, default=64); p.add_argument("--data-seed", type=int, default=0)
p.add_argument("--dense", type=int, default=10); p.add_argument("--sparse", type=int, default=50)
p.add_argument("--fine", type=int, default=5); p.add_argument("--match-every", type=int, default=2)
a = p.parse_args(); dev = "cuda"; torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

if a.task == "cifar":
    import pandas as pd
    from PIL import Image
    def load(split):
        df = pd.read_parquet(f"{a.data}/cifar10/{split}.parquet")
        x = np.stack([np.array(Image.open(io.BytesIO(b["bytes"]))) for b in df["img"]])
        return torch.tensor(x).permute(0, 3, 1, 2).contiguous(), torch.tensor(df["label"].values)
    Xtr, Ytr = load("train"); Xte, Yte = load("test")
    mean = Xtr.float().mean((0, 2, 3), keepdim=True); std = Xtr.float().std((0, 2, 3), keepdim=True)
    norm = lambda x: ((x.float() - mean.to(x.device)) / std.to(x.device))
    Xtr, Ytr, Xte, Yte = Xtr.to(dev), Ytr.to(dev), norm(Xte.to(dev)), Yte.to(dev)
    N = len(Xtr)

    def conv_bn(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.GELU())
    class Res(nn.Module):
        def __init__(s, c):
            super().__init__(); s.a = conv_bn(c, c); s.b = conv_bn(c, c)
        def forward(s, x): return x + s.b(s.a(x))
    w = a.width
    model = nn.Sequential(conv_bn(3, w), conv_bn(w, 2 * w), nn.MaxPool2d(2), Res(2 * w), conv_bn(2 * w, 4 * w), nn.MaxPool2d(2),
                          conv_bn(4 * w, 8 * w), nn.MaxPool2d(2), Res(8 * w), nn.AdaptiveMaxPool2d(1), nn.Flatten(),
                          nn.Linear(8 * w, 10, bias=False)).to(dev).to(memory_format=torch.channels_last)

    def batch(step):
        g = torch.Generator(device=dev).manual_seed(a.data_seed * 1_000_003 + step)
        ep, k = divmod(step * a.bs, N)
        perm = torch.randperm(N, device=dev, generator=torch.Generator(device=dev).manual_seed(a.data_seed * 7919 + ep))
        idx = perm[torch.arange(k, k + a.bs, device=dev) % N]
        x = norm(Xtr[idx])
        flip = torch.rand(a.bs, device=dev, generator=g) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        pad = F.pad(x, (4, 4, 4, 4))                                   # zero = mean colour after normalisation
        dx, dy = torch.randint(0, 9, (2,), device=dev, generator=g).tolist()   # one random crop offset per minibatch
        return pad[:, :, dy:dy + 32, dx:dx + 32].contiguous(memory_format=torch.channels_last), Ytr[idx]

    def lossf(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return F.cross_entropy(model(x).float(), y)

    @torch.no_grad()
    def evaluate():
        model.eval(); tot = 0.0
        for i in range(0, len(Xte), 1000):
            tot += lossf(Xte[i:i + 1000].contiguous(memory_format=torch.channels_last), Yte[i:i + 1000]).item() * 1000
        model.train(); return tot / len(Xte)

elif a.task == "mf":
    r = np.array([l.split("::")[:3] for l in open(f"{a.data}/ml-1m/ratings.dat", encoding="latin-1")], dtype=np.int64)
    u = np.unique(r[:, 0], return_inverse=True)[1]; it = np.unique(r[:, 1], return_inverse=True)[1]
    y = r[:, 2].astype(np.float32); mu = float(y.mean())
    sp = np.random.default_rng(12345).random(len(y)) < 0.1
    U, I, Yr = (torch.tensor(v[~sp], device=dev) for v in (u, it, y)); Ut, It, Yt = (torch.tensor(v[sp], device=dev) for v in (u, it, y))
    N = len(Yr)
    class MF(nn.Module):
        def __init__(s, nu, ni, d):
            super().__init__()
            s.P = nn.Embedding(nu, d); s.Q = nn.Embedding(ni, d); s.bu = nn.Embedding(nu, 1); s.bi = nn.Embedding(ni, 1)
            for e in (s.P, s.Q): nn.init.normal_(e.weight, 0, 0.1)
            for e in (s.bu, s.bi): nn.init.zeros_(e.weight)
        def forward(s, uu, ii):
            return mu + (s.P(uu) * s.Q(ii)).sum(1) + s.bu(uu)[:, 0] + s.bi(ii)[:, 0]
    model = MF(int(u.max()) + 1, int(it.max()) + 1, a.dim).to(dev)

    def batch(step):
        ep, k = divmod(step * a.bs, N)
        perm = torch.randperm(N, device=dev, generator=torch.Generator(device=dev).manual_seed(a.data_seed * 7919 + ep))
        idx = perm[torch.arange(k, k + a.bs, device=dev) % N]
        return (U[idx], I[idx]), Yr[idx]

    def lossf(x, y): return F.mse_loss(model(*x), y)

    @torch.no_grad()
    def evaluate(): return F.mse_loss(model(Ut, It), Yt).item()

params = list(model.parameters())
if a.opt == "sgd":
    opt = torch.optim.SGD(params, lr=0.1, momentum=a.mom, weight_decay=a.wd, nesterov=a.mom > 0)
else:
    opt = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), weight_decay=a.wd)
step0 = 0
if a.init:
    ck = torch.load(a.init, map_location=dev, weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    step0 = ck["step"]

ref_s, ref_l = np.array([]), np.array([])
if a.ref:
    rr = [line.rstrip("\n").split("\t") for line in open(a.ref)][1:]
    ref_s = np.array([int(f[0]) for f in rr]); ref_l = np.array([float(f[3]) for f in rr]); o = np.argsort(ref_s)
    ref_s, ref_l = ref_s[o], ref_l[o]
refv = lambda st: float(np.interp(st, ref_s, ref_l))
sched = json.loads(a.sched); save_at = {int(s) for s in a.save_at.split(",") if s}
os.makedirs(a.res, exist_ok=True); os.makedirs(a.ckdir, exist_ok=True)
log = open(f"{a.res}/{a.out}_eval.tsv", "w"); log.write("step\tphase\tlr\teval_loss\ttrain_loss\twnorm\twall\n")
meta = {"args": vars(a), "step0": step0, "phases": []}; t0 = time.time(); s = step0; acc, nacc = 0.0, 0


last = [None, None]

def record(ph, lr):
    global acc, nacc
    e = evaluate(); wn = float(torch.sqrt(sum((q.float() ** 2).sum() for q in params)))
    log.write(f"{s}\t{ph}\t{lr:.6g}\t{e:.6f}\t{acc / max(nacc, 1):.6f}\t{wn:.4f}\t{time.time() - t0:.1f}\n"); log.flush()
    acc, nacc = 0.0, 0; last[0], last[1] = s, e; return e


def rescale_to(path):
    """function-preserving rescaling: every conv filter that feeds a batch-norm layer is scaled, per output channel, to the norm
    it has in the checkpoint at `path`; the BN running mean (x c) and variance (x c^2) and the SGD momentum buffer (/ c) are
    adjusted so that only the weight norm, i.e. the effective learning rate of these scale-invariant filters, changes"""
    ref = torch.load(path, map_location=dev, weights_only=False)["model"]; names = {m: n for n, m in model.named_modules()}
    cs = []
    for mod in model.modules():
        if isinstance(mod, nn.Sequential) and len(mod) >= 2 and isinstance(mod[0], nn.Conv2d) and isinstance(mod[1], nn.BatchNorm2d):
            cv, bn = mod[0], mod[1]; w = cv.weight.data; wr = ref[names[cv] + ".weight"].to(w)
            c = wr.flatten(1).norm(dim=1) / w.flatten(1).norm(dim=1).clamp_min(1e-12)
            w.mul_(c.view(-1, 1, 1, 1)); bn.running_mean.mul_(c); bn.running_var.mul_(c ** 2)
            buf = opt.state.get(cv.weight, {}).get("momentum_buffer")
            if buf is not None: buf.div_(c.view(-1, 1, 1, 1))
            cs.append(c)
    c = torch.cat(cs); return {"rescaled_filters": int(len(c)), "scale_mean": float(c.mean()), "scale_min": float(c.min()),
                               "scale_max": float(c.max())}


record(-1, 0.0)
for ip, ph in enumerate(sched):
    if ph.get("rescale_from"):
        rs_info = rescale_to(ph["rescale_from"]); print(json.dumps(rs_info), flush=True); record(-2, 0.0)
    start = s; match = ph.get("until") == "match"; nmax = ph.get("max", 10 ** 9) if match else ph["steps"]
    sign0 = None; info = {"phase": ip, "start": start, "lr": ph["lr"], "match": match}
    if match and last[0] == s and len(ref_s) and ref_s[0] <= s <= ref_s[-1]:
        sign0 = np.sign(last[1] - refv(s)); info["gap_start"] = last[1] - refv(s)
    while s - start < nmax:
        lr = ph["lr"] * min(1.0, (s - start + 1) / ph["steps"]) if ph.get("warmup") else ph["lr"]
        for g in opt.param_groups: g["lr"] = lr
        x, y = batch(s); loss = lossf(x, y); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        s += 1; acc += loss.item(); nacc += 1; k = s - start
        every = a.match_every if match else (a.fine if k <= 100 else a.dense if k <= 300 else a.sparse)
        if (k % every) == 0 or k == nmax:
            e = record(ip, lr)
            if match:
                d = e - refv(s)
                if sign0 is None: sign0 = np.sign(d)
                elif np.sign(d) != sign0: info.update(matched_at=s, match_gap=d); break
        if s in save_at:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s}, f"{a.ckdir}/{a.out}_s{s}.pt")
    if ph.get("rescale_from"): info["rescale"] = rs_info
    if ph.get("save_end"):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s}, f"{a.ckdir}/{a.out}_end{ip}.pt")
    info["end"] = s; meta["phases"].append(info); print(json.dumps(info), flush=True)
meta["wall"] = time.time() - t0
json.dump(meta, open(f"{a.res}/{a.out}_meta.json", "w"), indent=1)
print("DONE", a.out, f"{meta['wall']:.0f}s", flush=True)
