#!/bin/bash
# E10 other learning machines: matrix-factorisation recommender (MovieLens-1M) and CIFAR-10 CNN, same protocol set.
# Usage: run_small.sh <mf|cifar>
task=$1
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_small.py --task $task"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
if [ $task = mf ]; then
  A="--opt sgd --mom 0.9 --wd 1e-4 --bs 4096"; E=1.0; H=3.0; Lo=0.125; NB=8000; W=200
else
  A="--opt sgd --mom 0.9 --wd 5e-4 --bs 256"; E=0.04; H=0.12; Lo=0.005; NB=4000; W=200
fi
KP() { echo "[{\"lr\":$1,\"steps\":$2},{\"lr\":$3,\"until\":\"match\",\"max\":3000},{\"lr\":$E,\"steps\":2500}]"; }
set -x
[ -f $C/${task}_base_s$NB.pt ] || $PY $A --out ${task}_base --sched "[{\"lr\":$E,\"steps\":$W,\"warmup\":1},{\"lr\":$E,\"steps\":$((NB-W))}]" --save-at $NB > $L/${task}_base.log 2>&1
I="--init $C/${task}_base_s$NB.pt"
[ -f $R/${task}_refL_meta.json ] || $PY $A $I --out ${task}_refL --sched "[{\"lr\":$E,\"steps\":5000}]" --dense 10 --sparse 10 > $L/${task}_refL.log 2>&1
[ -f $R/${task}_kp_meta.json ] || $PY $A $I --out ${task}_kp --ref $R/${task}_refL_eval.tsv --sched "$(KP $H 300 $Lo)" > $L/${task}_kp.log 2>&1
[ -f $R/${task}_km_meta.json ] || $PY $A $I --out ${task}_km --ref $R/${task}_refL_eval.tsv --sched "$(KP $Lo 300 $H)" > $L/${task}_km.log 2>&1
[ -f $R/${task}_pulse_hot_meta.json ] || $PY $A $I --out ${task}_pulse_hot --sched "[{\"lr\":$H,\"steps\":300},{\"lr\":$E,\"steps\":2700}]" > $L/${task}_pulse_hot.log 2>&1
[ -f $R/${task}_pulse_cold_meta.json ] || $PY $A $I --out ${task}_pulse_cold --sched "[{\"lr\":$Lo,\"steps\":300},{\"lr\":$E,\"steps\":2700}]" > $L/${task}_pulse_cold.log 2>&1
for lr in $Lo $H; do
  [ -f $R/${task}_step_${lr}_meta.json ] || $PY $A $I --out ${task}_step_$lr --sched "[{\"lr\":$lr,\"steps\":3000}]" > $L/${task}_step_$lr.log 2>&1
done
echo ALLDONE
