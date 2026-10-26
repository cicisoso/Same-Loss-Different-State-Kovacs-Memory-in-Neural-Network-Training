#!/bin/bash
# E10 weight-decay control (Codex eval #5): with decoupled weight decay 0 the weight norm no longer relaxes to a
# decay-set equilibrium, so a norm-mediated (effective-learning-rate) slow mode is removed. Same init and data order as base.
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512 --wd 0"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
while ! grep -q ALLDONE $L/run_small_chain.log 2>/dev/null; do sleep 30; done
KP() { echo "[{\"lr\":$1,\"steps\":$2},{\"lr\":$3,\"until\":\"match\",\"max\":3000},{\"lr\":1e-3,\"steps\":2500}]"; }
set -x
[ -f $C/base_wd0_s4000.pt ] || $PY --out base_wd0 --sched '[{"lr":1e-3,"steps":500,"warmup":1},{"lr":1e-3,"steps":3500}]' --save-at 4000 > $L/base_wd0.log 2>&1
I="--init $C/base_wd0_s4000.pt"
[ -f $R/refL_wd0_meta.json ] || $PY $I --out refL_wd0 --sched '[{"lr":1e-3,"steps":4500}]' --dense 10 --sparse 10 > $L/refL_wd0.log 2>&1
[ -f $R/kp_wd0_meta.json ] || $PY $I --out kp_wd0 --ref $R/refL_wd0_eval.tsv --sched "$(KP 3e-3 300 1.25e-4)" > $L/kp_wd0.log 2>&1
[ -f $R/km_wd0_meta.json ] || $PY $I --out km_wd0 --match-every 1 --ref $R/refL_wd0_eval.tsv --sched "$(KP 1.25e-4 300 1.5e-3)" > $L/km_wd0.log 2>&1
[ -f $R/pulse_hot_wd0_meta.json ] || $PY $I --out pulse_hot_wd0 --sched '[{"lr":3e-3,"steps":300},{"lr":1e-3,"steps":2700}]' > $L/pulse_hot_wd0.log 2>&1
[ -f $R/pulse_cold_wd0_meta.json ] || $PY $I --out pulse_cold_wd0 --sched '[{"lr":1.25e-4,"steps":300},{"lr":1e-3,"steps":2700}]' > $L/pulse_cold_wd0.log 2>&1
echo ALLDONE
