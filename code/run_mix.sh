#!/bin/bash
# E10 data-composition Kovacs ("spontaneous recovery"): web text (domain 1) + Python code (domain 2), lr fixed at 1e-3.
# Base and reference train on 25% code. kpm: code only for 300 steps (web forgotten), then web only until the web held-out
# loss matches the reference, then back to 25% code. kmm: web only 300 steps, then code only until matched, then 25%.
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512 --match-every 1"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov; D=/home/soso/advsci/data/kov
while ! grep -q ALLDONE $L/run_lin.log 2>/dev/null; do sleep 30; done
DA="--data2 $D/code --train-glob2 code_train_ --val-file2 code_val_000000.bin --mix 0.25"
set -x
[ -f $C/mix_base_s4000.pt ] || $PY $DA --out mix_base --sched '[{"lr":1e-3,"steps":500,"warmup":1},{"lr":1e-3,"steps":3500}]' --save-at 4000 > $L/mix_base.log 2>&1
I="--init $C/mix_base_s4000.pt"
[ -f $R/mix_refL_meta.json ] || $PY $DA $I --out mix_refL --sched '[{"lr":1e-3,"steps":4500}]' --dense 10 --sparse 10 > $L/mix_refL.log 2>&1
[ -f $R/mix_kp_meta.json ] || $PY $DA $I --out mix_kp --ref $R/mix_refL_eval.tsv --sched '[{"lr":1e-3,"steps":300,"mix":1.0},{"lr":1e-3,"until":"match","max":3000,"mix":0.0},{"lr":1e-3,"steps":2500}]' > $L/mix_kp.log 2>&1
[ -f $R/mix_km_meta.json ] || $PY $DA $I --out mix_km --ref $R/mix_refL_eval.tsv --sched '[{"lr":1e-3,"steps":300,"mix":0.0},{"lr":1e-3,"until":"match","max":3000,"mix":1.0},{"lr":1e-3,"steps":2500}]' > $L/mix_km.log 2>&1
[ -f $R/mix_pulse_code_meta.json ] || $PY $DA $I --out mix_pulse_code --sched '[{"lr":1e-3,"steps":300,"mix":1.0},{"lr":1e-3,"steps":2700}]' > $L/mix_pulse_code.log 2>&1
[ -f $R/mix_pulse_web_meta.json ] || $PY $DA $I --out mix_pulse_web --sched '[{"lr":1e-3,"steps":300,"mix":0.0},{"lr":1e-3,"steps":2700}]' > $L/mix_pulse_web.log 2>&1
echo ALLDONE
