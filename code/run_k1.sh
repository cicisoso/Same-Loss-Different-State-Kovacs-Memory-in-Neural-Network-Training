#!/bin/bash
# E10 pilot: GPT-30M on FineWeb, AdamW, reference lr 1e-3. Base run to step 4000, then from that checkpoint:
#   ref        constant 1e-3, held-out loss every 10 steps (matching target)
#   kp / km    Kovacs protocols: heat (3e-3, 300 steps) then cool (1.25e-4) until matched, release at 1e-3  (kp)
#              cool (1.25e-4, 300 steps) then heat (3e-3) until matched, release at 1e-3                    (km)
#   pulse_*    single 300-step pulses then 1e-3; step_*  constant lr from the checkpoint (response kernel)
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
set -x
[ -f $C/base_s4000.pt ] || $PY --out base --sched '[{"lr":1e-3,"steps":500,"warmup":1},{"lr":1e-3,"steps":3500}]' --save-at 4000 > $L/base.log 2>&1
I="--init $C/base_s4000.pt"
[ -f $R/ref_meta.json ] || $PY $I --out ref --sched '[{"lr":1e-3,"steps":3000}]' --dense 10 --sparse 10 > $L/ref.log 2>&1
[ -f $R/kp_meta.json ] || $PY $I --out kp --ref $R/ref_eval.tsv --sched '[{"lr":3e-3,"steps":300},{"lr":1.25e-4,"until":"match","max":1500},{"lr":1e-3,"steps":2500}]' > $L/kp.log 2>&1
[ -f $R/km_meta.json ] || $PY $I --out km --ref $R/ref_eval.tsv --sched '[{"lr":1.25e-4,"steps":300},{"lr":3e-3,"until":"match","max":1500},{"lr":1e-3,"steps":2500}]' > $L/km.log 2>&1
[ -f $R/pulse_hot_meta.json ] || $PY $I --out pulse_hot --sched '[{"lr":3e-3,"steps":300},{"lr":1e-3,"steps":2700}]' > $L/pulse_hot.log 2>&1
[ -f $R/pulse_cold_meta.json ] || $PY $I --out pulse_cold --sched '[{"lr":1.25e-4,"steps":300},{"lr":1e-3,"steps":2700}]' > $L/pulse_cold.log 2>&1
for lr in 1.25e-4 3e-3 2.5e-4 2e-3 5e-4; do
  [ -f $R/step_${lr}_meta.json ] || $PY $I --out step_$lr --sched "[{\"lr\":$lr,\"steps\":3000}]" > $L/step_$lr.log 2>&1
done
echo ALLDONE
