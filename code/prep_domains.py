# E10: tokenize two more text domains (Python code, PubMed abstracts) into the FineWeb .bin format used by kov_train.py
# (1024-byte header: int32 magic 20240520, version 1, token count; then uint16 GPT-2 BPE tokens, documents separated by <|endoftext|>).
# The first documents (>= 2 M tokens) form the validation shard; training shards hold 100 M tokens each.
# Usage: python -I prep_domains.py <code|pubmed> <raw_dir> <out_dir> <max_train_tokens>
import sys, os, gzip, json, numpy as np, tiktoken

dom, raw, out, cap = sys.argv[1], sys.argv[2], sys.argv[3], int(float(sys.argv[4]))
enc = tiktoken.get_encoding("gpt2"); EOT = enc.eot_token
os.makedirs(out, exist_ok=True)


def docs():
    for fn in sorted(os.listdir(raw)):
        p = os.path.join(raw, fn)
        if dom == "code" and fn.endswith(".json.gz"):
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    yield json.loads(line)["content"]
        elif dom == "pubmed" and fn.endswith(".jsonl"):
            with open(p) as fh:
                for line in fh:
                    d = json.loads(line); yield (d.get("title") or "") + "\n" + (d.get("content") or "")


def write(name, toks):
    a = np.asarray(toks, dtype=np.uint16); h = np.zeros(256, dtype=np.int32); h[0], h[1], h[2] = 20240520, 1, len(a)
    with open(os.path.join(out, name), "wb") as fh:
        fh.write(h.tobytes()); fh.write(a.tobytes())
    print(name, len(a), flush=True)


buf, val_done, shard, total, batch = [], False, 1, 0, []
def flush_batch():
    global buf
    for ids in enc.encode_ordinary_batch(batch, num_threads=32):
        buf.extend(ids); buf.append(EOT)
for text in docs():
    batch.append(text)
    if len(batch) < 2048: continue
    flush_batch(); batch = []
    if not val_done and len(buf) >= 2_000_000:
        write(f"{dom}_val_000000.bin", buf); buf = []; val_done = True
    while len(buf) >= 100_000_000:
        write(f"{dom}_train_{shard:06d}.bin", buf[:100_000_000]); buf = buf[100_000_000:]; total += 100_000_000; shard += 1
    if total >= cap: break
if batch: flush_batch()
if total < cap and len(buf) > 1_000_000: write(f"{dom}_train_{shard:06d}.bin", buf); total += len(buf)
print("DONE", dom, total, flush=True)
