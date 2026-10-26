#!/bin/bash
# E10 larger language model (Codex eval #7): GPT-124M (12 layers, d=768, 12 heads), same data/optimizer/protocols as the 30M pilot.
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512 --layers 12 --dim 768 --heads 12"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
while ! grep -q ALLDONE $L/run_swap.log 2>/dev/null; do sleep 30; done
KP() { echo "[{\"lr\":$1,\"steps\":$2},{\"lr\":$3,\"until\":\"match\",\"max\":3000},{\"lr\":1e-3,\"steps\":2500}]"; }
set -x
[ -f $C/big_base_s4000.pt ] || $PY --out big_base --sched '[{"lr":1e-3,"steps":500,"warmup":1},{"lr":1e-3,"steps":3500}]' --save-at 4000 > $L/big_base.log 2>&1
I="--init $C/big_base_s4000.pt"
[ -f $R/big_refL_meta.json ] || $PY $I --out big_refL --sched '[{"lr":1e-3,"steps":4500}]' --dense 10 --sparse 10 > $L/big_refL.log 2>&1
[ -f $R/big_kp_meta.json ] || $PY $I --out big_kp --match-every 2 --ref $R/big_refL_eval.tsv --sched "$(KP 3e-3 300 1.25e-4)" > $L/big_kp.log 2>&1
[ -f $R/big_pulse_hot_meta.json ] || $PY $I --out big_pulse_hot --sched '[{"lr":3e-3,"steps":300},{"lr":1e-3,"steps":2700}]' > $L/big_pulse_hot.log 2>&1
[ -f $R/big_pulse_cold_meta.json ] || $PY $I --out big_pulse_cold --sched '[{"lr":1.25e-4,"steps":300},{"lr":1e-3,"steps":2700}]' > $L/big_pulse_cold.log 2>&1
echo ALLDONE
