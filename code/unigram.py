# Unigram counts of a tokenised training stream (uint16 .bin shards with the 1024-byte header), used by kov_train.py to bin
# held-out target tokens by frequency rank. Usage: python -I unigram.py <data_dir> <train_glob> <out.npy> [max_shards]
import sys, os, numpy as np

d, g, out = sys.argv[1:4]; ms = int(sys.argv[4]) if len(sys.argv) > 4 else 2
fs = sorted(f for f in os.listdir(d) if f.startswith(g) and f.endswith(".bin"))[:ms]
c = np.zeros(65536, np.int64)
for f in fs:
    hdr = np.fromfile(os.path.join(d, f), dtype=np.int32, count=256); assert hdr[0] == 20240520, f
    v = np.memmap(os.path.join(d, f), dtype=np.uint16, mode="r", offset=1024, shape=(int(hdr[2]),))
    c += np.bincount(v, minlength=65536)
np.save(out, c[:50304]); print(out, fs, int(c.sum()), "nonzero", int((c > 0).sum()))
