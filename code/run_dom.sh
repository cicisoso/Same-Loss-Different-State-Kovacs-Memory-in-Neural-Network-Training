#!/bin/bash
# E10 domains: the core protocol set on Python code and PubMed abstracts (same GPT-30M, same lr scale).
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov; D=/home/soso/advsci/data/kov
KP() { echo "[{\"lr\":$1,\"steps\":$2},{\"lr\":$3,\"until\":\"match\",\"max\":3000},{\"lr\":1e-3,\"steps\":2500}]"; }
set -x
for dom in code pubmed; do
  DA="--data $D/$dom --train-glob ${dom}_train_ --val-file ${dom}_val_000000.bin"
  while [ -f $L/${dom}_base.log ] && [ ! -f $R/${dom}_base_meta.json ]; do sleep 30; done      # base running elsewhere
  [ -f $C/${dom}_base_s4000.pt ] || $PY $DA --out ${dom}_base --sched '[{"lr":1e-3,"steps":500,"warmup":1},{"lr":1e-3,"steps":3500}]' --save-at 4000 > $L/${dom}_base.log 2>&1
  I="--init $C/${dom}_base_s4000.pt"
  [ -f $R/${dom}_refL_meta.json ] || $PY $DA $I --out ${dom}_refL --sched '[{"lr":1e-3,"steps":4500}]' --dense 10 --sparse 10 > $L/${dom}_refL.log 2>&1
  [ -f $R/${dom}_kp_meta.json ] || $PY $DA $I --out ${dom}_kp --ref $R/${dom}_refL_eval.tsv --sched "$(KP 3e-3 300 1.25e-4)" > $L/${dom}_kp.log 2>&1
  [ -f $R/${dom}_km_meta.json ] || $PY $DA $I --out ${dom}_km --match-every 1 --ref $R/${dom}_refL_eval.tsv --sched "$(KP 1.25e-4 300 1.5e-3)" > $L/${dom}_km.log 2>&1
  [ -f $R/${dom}_pulse_hot_meta.json ] || $PY $DA $I --out ${dom}_pulse_hot --sched '[{"lr":3e-3,"steps":300},{"lr":1e-3,"steps":2700}]' > $L/${dom}_pulse_hot.log 2>&1
  [ -f $R/${dom}_pulse_cold_meta.json ] || $PY $DA $I --out ${dom}_pulse_cold --sched '[{"lr":1.25e-4,"steps":300},{"lr":1e-3,"steps":2700}]' > $L/${dom}_pulse_cold.log 2>&1
  for lr in 1.25e-4 3e-3; do
    [ -f $R/${dom}_step_${lr}_meta.json ] || $PY $DA $I --out ${dom}_step_$lr --sched "[{\"lr\":$lr,\"steps\":3000}]" > $L/${dom}_step_$lr.log 2>&1
  done
done
echo ALLDONE
