#!/bin/bash
# E10 other learning machines, parametrised: run_small2.sh <task: mf|cifar> <tag> <eta_ref> <eta_hot> <eta_cold> <eta_match_hot>
#   <base_steps> "<extra kov_small args>"  — base, reference, heat->cool Kovacs, cool->heat Kovacs (gentler heating, 1-step
#   matching), single pulses and constant steps; waits for the stream given in WAIT (log file containing ALLDONE) if set.
task=$1; T=$2; E=$3; H=$4; Lo=$5; HM=$6; NB=$7; A=$8
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_small.py --task $task"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
[ -n "$WAIT" ] && while ! grep -q ALLDONE $WAIT 2>/dev/null; do sleep 30; done
KP() { echo "[{\"lr\":$1,\"steps\":$2},{\"lr\":$3,\"until\":\"match\",\"max\":3000},{\"lr\":$E,\"steps\":2500}]"; }
set -x
[ -f $C/${T}_base_s$NB.pt ] || $PY $A --out ${T}_base --sched "[{\"lr\":$E,\"steps\":200,\"warmup\":1},{\"lr\":$E,\"steps\":$((NB-200))}]" --save-at $NB > $L/${T}_base.log 2>&1
I="--init $C/${T}_base_s$NB.pt"
[ -f $R/${T}_refL_meta.json ] || $PY $A $I --out ${T}_refL --sched "[{\"lr\":$E,\"steps\":5000}]" --dense 10 --sparse 10 > $L/${T}_refL.log 2>&1
[ -f $R/${T}_kp_meta.json ] || $PY $A $I --out ${T}_kp --match-every 1 --ref $R/${T}_refL_eval.tsv --sched "$(KP $H 300 $Lo)" > $L/${T}_kp.log 2>&1
[ -f $R/${T}_km_meta.json ] || $PY $A $I --out ${T}_km --match-every 1 --ref $R/${T}_refL_eval.tsv --sched "$(KP $Lo 300 $HM)" > $L/${T}_km.log 2>&1
[ -f $R/${T}_pulse_hot_meta.json ] || $PY $A $I --out ${T}_pulse_hot --sched "[{\"lr\":$H,\"steps\":300},{\"lr\":$E,\"steps\":2700}]" > $L/${T}_pulse_hot.log 2>&1
[ -f $R/${T}_pulse_cold_meta.json ] || $PY $A $I --out ${T}_pulse_cold --sched "[{\"lr\":$Lo,\"steps\":300},{\"lr\":$E,\"steps\":2700}]" > $L/${T}_pulse_cold.log 2>&1
for lr in $Lo $H; do
  [ -f $R/${T}_step_${lr}_meta.json ] || $PY $A $I --out ${T}_step_$lr --sched "[{\"lr\":$lr,\"steps\":3000}]" > $L/${T}_step_$lr.log 2>&1
done
echo ALLDONE
