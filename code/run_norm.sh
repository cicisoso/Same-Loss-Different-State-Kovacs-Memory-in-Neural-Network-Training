#!/bin/bash
# E10 norm interventions at the match (PREREG addendum 2026-10-20 06:51). Chain A: Kovacs branches; chain B: reference
# branches. All start from the matched checkpoints written by run_swap.sh (kpS_end1.pt and refS_s<TM>.pt) and run 1500 steps at
# eta_ref with the same data and angular-update logging.
# Usage: bash run_norm.sh A|B
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512 --log-angle 1"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
while [ ! -f $R/refS_meta.json ]; do sleep 60; done
TM=$(/home/soso/advsci/mlip/bin/python -I -c "import json,sys; print([p for p in json.load(open(sys.argv[1]))['phases'] if p.get('matched_at')][0]['matched_at'])" $R/kpS_meta.json)
K=$C/kpS_end1.pt; F=$C/refS_s$TM.pt
S='[{"lr":1e-3,"steps":1500}]'; E="--dense 10 --sparse 10"
set -x
run() { local out=$1; shift; [ -f $R/${out}_meta.json ] || $PY "$@" --out $out --sched "$S" $E > $L/$out.log 2>&1; }
if [ "$1" = A ]; then
  run nm_kp_sham --init $K
  run nm_kp_proj --init $K --rescale-proj to:$F
  run nm_kp_resid --init $K --rescale-resid to:$F
  run nm_kp_both --init $K --rescale-resid to:$F --rescale-proj to:$F
else
  run nm_ref_sham --init $F
  run nm_ref_proj --init $F --rescale-proj to:$K
  run nm_ref_resid --init $F --rescale-resid to:$K
fi
echo ALLDONE
