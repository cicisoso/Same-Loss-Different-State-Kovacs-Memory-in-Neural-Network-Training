#!/bin/bash
# E10 independent replicates (Codex eval: new minibatches from one C0 do not replicate initialization/history): five fresh
# initialisations with their own data order, each with base (4000 steps), reference, heat->cool Kovacs and cool->heat Kovacs.
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
while ! grep -q ALLDONE $L/run_k1.log 2>/dev/null; do sleep 30; done   # starts after the step runs, ahead of the domain queue
KP() { echo "[{\"lr\":$1,\"steps\":$2},{\"lr\":$3,\"until\":\"match\",\"max\":3000},{\"lr\":1e-3,\"steps\":2500}]"; }
set -x
for sd in 1 2 3 4 5; do                       # heat->cool for every seed first (slow approach: 2-step match evals)
  A="--seed $sd --data-seed $sd"
  [ -f $C/base_sd${sd}_s4000.pt ] || $PY $A --out base_sd$sd --sched '[{"lr":1e-3,"steps":500,"warmup":1},{"lr":1e-3,"steps":3500}]' --save-at 4000 > $L/base_sd$sd.log 2>&1
  I="--init $C/base_sd${sd}_s4000.pt"
  [ -f $R/refL_sd${sd}_meta.json ] || $PY $A $I --out refL_sd$sd --sched '[{"lr":1e-3,"steps":4500}]' --dense 10 --sparse 10 > $L/refL_sd$sd.log 2>&1
  [ -f $R/kp_sd${sd}_meta.json ] || $PY $A $I --out kp_sd$sd --match-every 2 --ref $R/refL_sd${sd}_eval.tsv --sched "$(KP 3e-3 300 1.25e-4)" > $L/kp_sd$sd.log 2>&1
done
for sd in 1 2 3 4 5; do                       # then cool->heat (fast approach: 1-step match evals)
  A="--seed $sd --data-seed $sd"; I="--init $C/base_sd${sd}_s4000.pt"
  [ -f $R/km_sd${sd}_meta.json ] || $PY $A $I --out km_sd$sd --match-every 1 --ref $R/refL_sd${sd}_eval.tsv --sched "$(KP 1.25e-4 300 1.5e-3)" > $L/km_sd$sd.log 2>&1
done
echo ALLDONE
