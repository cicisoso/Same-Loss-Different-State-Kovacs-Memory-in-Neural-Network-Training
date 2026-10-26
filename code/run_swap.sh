#!/bin/bash
# E10 optimizer-state swap control (suggested by the Codex review): Adam moments have ~10/20-step memories, comparable to the
# 30-60-step hump peak. Save the Kovacs run (kp protocol) and the reference at the matched step, then continue four runs at 1e-3
# with identical data: each set of weights with its own optimizer state and with the other run's state.
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
while [ ! -f $R/refL_meta.json ]; do sleep 30; done
I="--init $C/base_s4000.pt"
set -x
[ -f $C/kpS_end1.pt ] || $PY $I --out kpS --match-every 1 --ref $R/refL_eval.tsv --sched '[{"lr":3e-3,"steps":300},{"lr":1.25e-4,"until":"match","max":3000,"save_end":1}]' > $L/kpS.log 2>&1
TM=$(/home/soso/advsci/mlip/bin/python -I -c "import json,sys; print([p for p in json.load(open(sys.argv[1]))['phases'] if p.get('matched_at')][0]['matched_at'])" $R/kpS_meta.json)
[ -f $C/refS_s$TM.pt ] || $PY $I --out refS --sched "[{\"lr\":1e-3,\"steps\":$((TM-4000))}]" --save-at $TM > $L/refS.log 2>&1
S='[{"lr":1e-3,"steps":1500}]'; E="--dense 10 --sparse 10"
[ -f $R/sw_kp_own_meta.json ] || $PY --init $C/kpS_end1.pt --out sw_kp_own --sched "$S" $E > $L/sw_kp_own.log 2>&1
[ -f $R/sw_kp_refopt_meta.json ] || $PY --init $C/kpS_end1.pt --init-opt $C/refS_s$TM.pt --out sw_kp_refopt --sched "$S" $E > $L/sw_kp_refopt.log 2>&1
[ -f $R/sw_ref_own_meta.json ] || $PY --init $C/refS_s$TM.pt --out sw_ref_own --sched "$S" $E > $L/sw_ref_own.log 2>&1
[ -f $R/sw_ref_kpopt_meta.json ] || $PY --init $C/refS_s$TM.pt --init-opt $C/kpS_end1.pt --out sw_ref_kpopt --sched "$S" $E > $L/sw_ref_kpopt.log 2>&1
# consequence: anneal (linear decay to 0 over 500 steps) after B extra steps at 1e-3, from both matched checkpoints
for B in 0 200 1000; do
  for w in kp ref; do
    if [ $w = kp ]; then CK=$C/kpS_end1.pt; else CK=$C/refS_s$TM.pt; fi
    if [ $B -gt 0 ]; then SC="[{\"lr\":1e-3,\"steps\":$B},{\"lr\":1e-3,\"steps\":500,\"decay\":1}]"; else SC='[{"lr":1e-3,"steps":500,"decay":1}]'; fi
    [ -f $R/an_${w}_B${B}_meta.json ] || $PY --init $CK --out an_${w}_B$B --sched "$SC" --dense 10 --sparse 10 > $L/an_${w}_B$B.log 2>&1
  done
done
echo ALLDONE
