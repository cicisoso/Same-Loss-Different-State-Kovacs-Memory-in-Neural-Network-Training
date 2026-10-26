#!/bin/bash
# E10 stage 2 (FineWeb, after run_k1): longer reference, protocol variants (unseen pulse lengths and temperatures),
# optimizer-reset control, the reverse protocol with gentler heating (1.5e-3; 3e-3 overshoots the match within 2 steps),
# and two replicates on unseen data (stream offset 5000 / 10000 steps).
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
while ! grep -q ALLDONE $L/run_k1.log; do sleep 30; done
set -x
I="--init $C/base_s4000.pt"
KP() { echo "[{\"lr\":$1,\"steps\":$2},{\"lr\":$3,\"until\":\"match\",\"max\":3000},{\"lr\":1e-3,\"steps\":2500${4}}]"; }
[ -f $R/refL_meta.json ] || $PY $I --out refL --sched '[{"lr":1e-3,"steps":4500}]' --dense 10 --sparse 10 --save-at 8000 > $L/refL.log 2>&1
[ -f $R/kp_ta100_meta.json ] || $PY $I --out kp_ta100 --ref $R/refL_eval.tsv --sched "$(KP 3e-3 100 1.25e-4)" > $L/kp_ta100.log 2>&1
[ -f $R/kp_ta1000_meta.json ] || $PY $I --out kp_ta1000 --ref $R/refL_eval.tsv --sched "$(KP 3e-3 1000 1.25e-4)" > $L/kp_ta1000.log 2>&1
[ -f $R/kp_h2_meta.json ] || $PY $I --out kp_h2 --ref $R/refL_eval.tsv --sched "$(KP 2e-3 300 1.25e-4)" > $L/kp_h2.log 2>&1
[ -f $R/kp_l2_meta.json ] || $PY $I --out kp_l2 --ref $R/refL_eval.tsv --sched "$(KP 3e-3 300 2.5e-4)" > $L/kp_l2.log 2>&1
[ -f $R/kp_reset_meta.json ] || $PY $I --out kp_reset --ref $R/refL_eval.tsv --sched "$(KP 3e-3 300 1.25e-4 ',"reset":1')" > $L/kp_reset.log 2>&1
TM=$(/home/soso/advsci/mlip/bin/python -I -c "import json,sys; print([p for p in json.load(open(sys.argv[1]))['phases'] if p.get('matched_at')][0]['matched_at'])" $R/kp_reset_meta.json)
[ -f $R/refL_reset_meta.json ] || $PY $I --out refL_reset --sched "[{\"lr\":1e-3,\"steps\":$((TM-4000))},{\"lr\":1e-3,\"steps\":$((8500-TM)),\"reset\":1}]" --dense 10 --sparse 10 > $L/refL_reset.log 2>&1
[ -f $R/km2_meta.json ] || $PY $I --out km2 --match-every 1 --ref $R/refL_eval.tsv --sched "$(KP 1.25e-4 300 1.5e-3)" > $L/km2.log 2>&1
for r in 1 2; do
  O="--data-offset $((5000*r))"
  [ -f $R/refL_r${r}_meta.json ] || $PY $I $O --out refL_r$r --sched '[{"lr":1e-3,"steps":4500}]' --dense 10 --sparse 10 > $L/refL_r$r.log 2>&1
  [ -f $R/kp_r${r}_meta.json ] || $PY $I $O --out kp_r$r --ref $R/refL_r${r}_eval.tsv --sched "$(KP 3e-3 300 1.25e-4)" > $L/kp_r$r.log 2>&1
done
echo ALLDONE
