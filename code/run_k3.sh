#!/bin/bash
# E10 "calorimetry" probes of the matched state: at the Kovacs match, quench to 1.25e-4 for 400 steps; the reference is quenched
# at the same step. If fast modes of the Kovacs run are already cold (kp) its loss falls less than the reference's; after km
# (fast modes hot) it falls more. Runs after the domain queue (stream B).
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
while ! grep -q ALLDONE $L/run_dom.log 2>/dev/null; do sleep 30; done
while [ ! -f $R/refL_meta.json ]; do sleep 30; done
I="--init $C/base_s4000.pt"
TMF() { /home/soso/advsci/mlip/bin/python -I -c "import json,sys; print([p for p in json.load(open(sys.argv[1]))['phases'] if p.get('matched_at')][0]['matched_at'])" $1; }
set -x
for k in kp km; do
  if [ $k = kp ]; then A=3e-3; B=1.25e-4; else A=1.25e-4; B=1.5e-3; fi
  [ -f $R/${k}_probe_meta.json ] || $PY $I --out ${k}_probe --match-every 1 --ref $R/refL_eval.tsv --sched "[{\"lr\":$A,\"steps\":300},{\"lr\":$B,\"until\":\"match\",\"max\":3000},{\"lr\":1.25e-4,\"steps\":400}]" > $L/${k}_probe.log 2>&1
  TM=$(TMF $R/${k}_probe_meta.json)
  [ -f $R/ref_probe_${k}_meta.json ] || $PY $I --out ref_probe_$k --sched "[{\"lr\":1e-3,\"steps\":$((TM-4000))},{\"lr\":1.25e-4,\"steps\":400}]" > $L/ref_probe_$k.log 2>&1
done
echo ALLDONE
