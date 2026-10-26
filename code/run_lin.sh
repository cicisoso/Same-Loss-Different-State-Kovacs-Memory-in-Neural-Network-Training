#!/bin/bash
# E10 linear response and ageing (Codex eval #3): small up/down steps and pulses (+-20%), a two-pulse superposition test, and the
# heat->cool Kovacs protocol at a later training age (checkpoint at step 8000 of the reference).
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
while ! grep -q ALLDONE $L/run_k2.log 2>/dev/null; do sleep 30; done
I="--init $C/base_s4000.pt"
set -x
for lr in 1.2e-3 8e-4; do
  [ -f $R/step_${lr}_meta.json ] || $PY $I --out step_$lr --sched "[{\"lr\":$lr,\"steps\":3000}]" > $L/step_$lr.log 2>&1
  [ -f $R/pulse_${lr}_meta.json ] || $PY $I --out pulse_$lr --sched "[{\"lr\":$lr,\"steps\":300},{\"lr\":1e-3,\"steps\":2700}]" > $L/pulse_$lr.log 2>&1
done
[ -f $R/pulse_two_meta.json ] || $PY $I --out pulse_two --sched '[{"lr":3e-3,"steps":300},{"lr":1e-3,"steps":300},{"lr":1.25e-4,"steps":300},{"lr":1e-3,"steps":2100}]' > $L/pulse_two.log 2>&1
[ -f $R/pulse_cold_late_meta.json ] || $PY $I --out pulse_cold_late --sched '[{"lr":1e-3,"steps":600},{"lr":1.25e-4,"steps":300},{"lr":1e-3,"steps":2100}]' > $L/pulse_cold_late.log 2>&1
I8="--init $C/refL_s8000.pt"
[ -f $R/refA8_meta.json ] || $PY $I8 --out refA8 --sched '[{"lr":1e-3,"steps":4500}]' --dense 10 --sparse 10 > $L/refA8.log 2>&1
[ -f $R/kpA8_meta.json ] || $PY $I8 --out kpA8 --match-every 1 --ref $R/refA8_eval.tsv --sched '[{"lr":3e-3,"steps":300},{"lr":1.25e-4,"until":"match","max":3000},{"lr":1e-3,"steps":2500}]' > $L/kpA8.log 2>&1
echo ALLDONE
