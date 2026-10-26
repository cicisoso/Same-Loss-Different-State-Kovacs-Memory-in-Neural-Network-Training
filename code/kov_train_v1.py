# E10 "Kovacs memory in learning machines": GPT training under piecewise-constant learning-rate protocols.
# Every run that starts from the same checkpoint sees the same minibatch at the same step (data order fixed by --data-seed),
# so two runs differ only in their learning-rate history. Held-out loss is measured on a fixed validation set.
# Schedule (--sched, JSON list of phases, steps counted from the start checkpoint):
#   {"lr": x, "steps": n}                          constant lr for n steps (warmup: {"lr": x, "steps": n, "warmup": 1})
#   {"lr": x, "steps": n, "reset": 1}               as above, clearing the optimizer moments at the start of the phase
#   {"lr": x, "until": "match", "max": n}          constant lr until held-out loss crosses the reference run's loss at the
#                                                  same step (reference eval log given by --ref); the crossing is the Kovacs
#                                                  "matched" state
#   {"lr": x, "steps": n, "decay": 1}             linear decay from x to 0 over n steps (annealing)
#   {..., "save_end": 1}                           save model + optimizer at the end of the phase (e.g. at the match)
#   {"mix": f, ...}                                with --data2: fraction f of every minibatch drawn from the second domain
#                                                  (data-composition protocols; both streams advance by bs per step)
# Usage: python -I kov_train.py --out <tag> --sched '<json>' [--init ckpt.pt] [--save-at s1,s2] [--ref ref_eval.tsv] ...
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--out", required=True); p.add_argument("--sched", required=True)
p.add_argument("--init", default=""); p.add_argument("--save-at", default=""); p.add_argument("--ref", default="")
p.add_argument("--init-opt", default="", help="take the optimizer state from this checkpoint instead (state-swap control)")
p.add_argument("--data", default="/home/soso/advsci/data/kov/fineweb"); p.add_argument("--train-glob", default="fineweb_train_")
p.add_argument("--val-file", default="fineweb_val_000000.bin")
p.add_argument("--res", default="/home/soso/advsci/results/kov"); p.add_argument("--ckdir", default="/home/soso/advsci/ckpt/kov")
p.add_argument("--layers", type=int, default=6); p.add_argument("--dim", type=int, default=384); p.add_argument("--heads", type=int, default=6)
p.add_argument("--ctx", type=int, default=512); p.add_argument("--bs", type=int, default=64)
p.add_argument("--opt", default="adamw"); p.add_argument("--wd", type=float, default=0.1)
p.add_argument("--beta1", type=float, default=0.9); p.add_argument("--beta2", type=float, default=0.95)
p.add_argument("--clip", type=float, default=1.0); p.add_argument("--data-seed", type=int, default=0)
p.add_argument("--data-offset", type=int, default=0, help="read the data stream this many steps ahead (unseen replicate data)")
p.add_argument("--seed", type=int, default=0); p.add_argument("--eval-seqs", type=int, default=1024)
p.add_argument("--dense", type=int, default=10, help="eval interval inside the first 300 steps of a phase and in match phases")
p.add_argument("--sparse", type=int, default=50); p.add_argument("--reset-opt", type=int, default=0)
p.add_argument("--match-every", type=int, default=2, help="eval interval while matching (reference is interpolated)")
p.add_argument("--fine", type=int, default=5, help="eval interval inside the first 100 steps of a phase")
p.add_argument("--compile", type=int, default=1)
p.add_argument("--data2", default=""); p.add_argument("--train-glob2", default=""); p.add_argument("--val-file2", default="")
p.add_argument("--mix", type=float, default=0.0, help="default fraction of domain 2 per minibatch")
p.add_argument("--match-on", type=int, default=1, help="held-out set (1 or 2) whose loss is matched")
p.add_argument("--eval-b", type=int, default=1, help="also evaluate an untouched held-out set B (next eval-seqs sequences)")
a = p.parse_args()
torch.manual_seed(a.seed); dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True


# ---------------- data
def load_bin(path):
    hdr = np.fromfile(path, dtype=np.int32, count=256)
    assert hdr[0] == 20240520, path
    return np.memmap(path, dtype=np.uint16, mode="r", offset=1024, shape=(int(hdr[2]),))

shards = sorted(f for f in os.listdir(a.data) if f.startswith(a.train_glob) and f.endswith(".bin"))
train = [load_bin(os.path.join(a.data, f)) for f in shards]
L = a.ctx + 1
seq_index = np.concatenate([np.stack([np.full(len(t) // L, k), np.arange(len(t) // L) * L], 1) for k, t in enumerate(train)])
order = np.random.default_rng(a.data_seed).permutation(len(seq_index))

if a.data2:
    shards2 = sorted(f for f in os.listdir(a.data2) if f.startswith(a.train_glob2) and f.endswith(".bin"))
    train2 = [load_bin(os.path.join(a.data2, f)) for f in shards2]
    seq_index2 = np.concatenate([np.stack([np.full(len(t) // L, k), np.arange(len(t) // L) * L], 1) for k, t in enumerate(train2)])
    order2 = np.random.default_rng(a.data_seed).permutation(len(seq_index2))

def batch(step, mix=0.0):
    n2 = int(round(a.bs * mix)) if a.data2 else 0; base = (step + a.data_offset) * a.bs
    idx = seq_index[order[(base + np.arange(a.bs - n2)) % len(order)]]
    x = [train[k][o:o + L] for k, o in idx]
    if n2:
        x += [train2[k][o:o + L] for k, o in seq_index2[order2[(base + np.arange(n2)) % len(order2)]]]
    x = torch.from_numpy(np.stack(x).astype(np.int64)).pin_memory().to(dev, non_blocking=True)
    return x[:, :-1], x[:, 1:]

def load_eval(d, f):
    v = load_bin(os.path.join(d, f)); return torch.from_numpy(np.array(v[:a.eval_seqs * L]).astype(np.int64).reshape(a.eval_seqs, L))
ev = load_eval(a.data, a.val_file)
if a.eval_b:
    vb = load_bin(os.path.join(a.data, a.val_file))
    evb = torch.from_numpy(np.array(vb[a.eval_seqs * L: 2 * a.eval_seqs * L]).astype(np.int64).reshape(a.eval_seqs, L))
ev2 = load_eval(a.data2, a.val_file2) if a.data2 else None


# ---------------- model (GPT-2 style, pre-LN, tied embeddings, no dropout)
class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h
        s.ln1 = nn.LayerNorm(d, bias=False); s.qkv = nn.Linear(d, 3 * d, bias=False); s.proj = nn.Linear(d, d, bias=False)
        s.ln2 = nn.LayerNorm(d, bias=False); s.fc = nn.Linear(d, 4 * d, bias=False); s.out = nn.Linear(4 * d, d, bias=False)
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, 2)
        q, k, v = (t.view(B, T, s.h, C // s.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + s.proj(y)
        return x + s.out(F.gelu(s.fc(s.ln2(x))))

class GPT(nn.Module):
    def __init__(s, V, T, n, d, h):
        super().__init__()
        s.wte = nn.Embedding(V, d); s.wpe = nn.Embedding(T, d)
        s.blocks = nn.ModuleList([Block(d, h) for _ in range(n)]); s.lnf = nn.LayerNorm(d, bias=False)
        s.apply(s._init)
        for nme, prm in s.named_parameters():
            if nme.endswith("proj.weight") or nme.endswith("out.weight"):
                nn.init.normal_(prm, 0.0, 0.02 / math.sqrt(2 * n))
    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(s, idx):
        x = s.wte(idx) + s.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in s.blocks: x = b(x)
        return F.linear(s.lnf(x), s.wte.weight)

model = GPT(50304, a.ctx, a.layers, a.dim, a.heads).to(dev)
decay = [q for q in model.parameters() if q.dim() >= 2]; nodecay = [q for q in model.parameters() if q.dim() < 2]
groups = [{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}]
if a.opt == "adamw":
    opt = torch.optim.AdamW(groups, lr=1e-3, betas=(a.beta1, a.beta2), eps=1e-8, fused=True)
else:                                                             # momentum SGD with L2 weight decay
    opt = torch.optim.SGD(groups, lr=1e-3, momentum=a.beta1)
step0 = 0
if a.init:
    ck = torch.load(a.init, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); step0 = ck["step"]
    if a.init_opt: ck = torch.load(a.init_opt, map_location=dev, weights_only=False)
    if not a.reset_opt and ck.get("opt_kind", "adamw") == a.opt: opt.load_state_dict(ck["opt"])
class LM(nn.Module):                                              # loss inside the compiled graph: fused fp32 cross-entropy
    def __init__(s, m):
        super().__init__(); s.m = m
    def forward(s, x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = s.m(x)
        return F.cross_entropy(lg.float().view(-1, lg.size(-1)), y.reshape(-1))

lossf = torch.compile(LM(model)) if a.compile else LM(model)
nparam = sum(q.numel() for q in model.parameters())

@torch.no_grad()
def evaluate(E=None):
    E = ev if E is None else E
    model.eval(); tot = 0.0
    for i in range(0, a.eval_seqs, 64):
        x = E[i:i + 64].to(dev); tot += lossf(x[:, :-1], x[:, 1:]).item() * len(x)
    model.train(); return tot / a.eval_seqs

@torch.no_grad()
def norms():
    return (math.sqrt(sum(float(q.float().pow(2).sum()) for q in decay)),
            math.sqrt(sum(float(q.float().pow(2).sum()) for q in nodecay)))


ref_s, ref_l = [], []
if a.ref:
    for line in open(a.ref):
        f = line.rstrip("\n").split("\t")
        if f[0] == "step" or f[3] == "": continue
        ref_s.append(int(f[0])); ref_l.append(float(f[3] if a.match_on == 1 else f[9]))
    o = np.argsort(ref_s); ref_s, ref_l = np.array(ref_s)[o], np.array(ref_l)[o]
sched = json.loads(a.sched); save_at = {int(s) for s in a.save_at.split(",") if s}
os.makedirs(a.res, exist_ok=True); os.makedirs(a.ckdir, exist_ok=True)
log = open(f"{a.res}/{a.out}_eval.tsv", "w")
log.write("step\tphase\tlr\teval_loss\ttrain_loss\tgnorm\twnorm\tgain_norm\twall\teval_loss2\teval_loss_b\n")
tlog = open(f"{a.res}/{a.out}_train.tsv", "w"); tlog.write("step\tloss\n")                # per-step minibatch loss
meta = {"args": vars(a), "nparam": nparam, "step0": step0, "phases": []}
t_start = time.time(); s = step0; tl_acc, gn_acc, n_acc = 0.0, 0.0, 0


last = [None, None]                                                  # (step, matched-set loss) of the latest eval

def record(phase, lr, do_eval=True):
    global tl_acc, gn_acc, n_acc
    e = evaluate() if do_eval else float("nan"); w, g = norms()
    e2 = evaluate(ev2) if (do_eval and ev2 is not None) else float("nan")
    eb = evaluate(evb) if (do_eval and a.eval_b) else float("nan")
    tl = tl_acc / max(n_acc, 1); gn = gn_acc / max(n_acc, 1); tl_acc, gn_acc, n_acc = 0.0, 0.0, 0
    log.write(f"{s}\t{phase}\t{lr:.6g}\t{e:.6f}\t{tl:.6f}\t{gn:.5f}\t{w:.4f}\t{g:.4f}\t{time.time() - t_start:.1f}\t{e2:.6f}\t{eb:.6f}\n")
    log.flush()
    last[0], last[1] = s, (e2 if a.match_on == 2 else e)
    return last[1]


def save(tag):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": s, "opt_kind": a.opt}, f"{a.ckdir}/{tag}.pt")


record(-1, 0.0)
for ip, ph in enumerate(sched):
    start = s; match = ph.get("until") == "match"; nmax = ph.get("max", 10 ** 9) if match else ph["steps"]
    sign0 = None; info = {"phase": ip, "start": start, "lr": ph["lr"], "match": match}
    if ph.get("reset"): opt.state.clear(); info["reset"] = 1                 # fresh optimizer moments at this phase
    if match and last[0] == s and len(ref_s) and ref_s[0] <= s <= ref_s[-1]:   # side of the reference before the phase
        sign0 = np.sign(last[1] - float(np.interp(s, ref_s, ref_l))); info["gap_start"] = float(last[1] - np.interp(s, ref_s, ref_l))
    while s - start < nmax:
        lr = ph["lr"] * min(1.0, (s - start + 1) / ph["steps"]) if ph.get("warmup") else ph["lr"]
        if ph.get("decay"): lr = ph["lr"] * max(0.0, 1.0 - (s - start) / ph["steps"])     # linear anneal to zero
        for g in opt.param_groups: g["lr"] = lr
        x, y = batch(s, ph.get("mix", a.mix))
        loss = lossf(x, y); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip if a.clip > 0 else 1e9)
        opt.step(); opt.zero_grad(set_to_none=True)
        s += 1; lv = loss.item(); tl_acc += lv; gn_acc += float(gn); n_acc += 1; tlog.write(f"{s}\t{lv:.5f}\n")
        k = s - start
        every = a.match_every if match else (a.fine if k <= 100 else a.dense if k <= 300 else a.sparse)
        if (k % every) == 0 or k == nmax:
            e = record(ip, lr)
            if match:
                if not (ref_s[0] <= s <= ref_s[-1]): raise SystemExit(f"reference does not cover step {s}")
                d = e - float(np.interp(s, ref_s, ref_l))
                if sign0 is None: sign0 = np.sign(d)
                elif np.sign(d) != sign0:
                    info.update(matched_at=s, match_gap=d); break
        if s in save_at: save(f"{a.out}_s{s}")
    if ph.get("save_end"): save(f"{a.out}_end{ip}"); info["saved"] = f"{a.out}_end{ip}"
    info["end"] = s; meta["phases"].append(info)
    print(json.dumps(info), flush=True)
meta["wall"] = time.time() - t_start; tlog.close()
json.dump(meta, open(f"{a.res}/{a.out}_meta.json", "w"), indent=1)
print("DONE", a.out, f"{meta['wall']:.0f}s", flush=True)
