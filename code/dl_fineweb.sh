#!/bin/bash
# Pre-tokenized FineWeb-10B (GPT-2 BPE, uint16, 1024-byte header) shards used by modded-nanogpt; HF dataset kjj0/fineweb10B-gpt2.
D=/home/soso/advsci/data/kov/fineweb
cd $D
for f in fineweb_val_000000.bin $(for k in 1 2 3 4 5 6 7 8; do printf "fineweb_train_%06d.bin " $k; done); do
  [ -s $f ] && [ $(stat -c %s $f) -ge 100000000 ] && continue
  curl -sSL --retry 5 -o $f.part "https://huggingface.co/datasets/kjj0/fineweb10B-gpt2/resolve/main/$f" && mv $f.part $f
  echo "$f $(stat -c %s $f)"
done
echo DONE
