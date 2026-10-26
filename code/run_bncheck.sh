#!/bin/bash
# Exploratory check suggested by the third review (not part of the registered plan): does the lag of the batch-norm running
# statistics (momentum 0.1) contribute to the negative CIFAR excursion? The matched pair of the filter-norm exchange
# (cifar_kpS at its match, cifar_refS at the same step) is continued once more exactly like cifar_cn_own / cifar_cn_ref, with
# kov_small_bn.py, which also scores every evaluation after re-estimating the running statistics over 20 fixed minibatches
# (training untouched). Waits for the plain-SGD CIFAR chain (run_cifar0.log) so that its GPU memory is reused.
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_small_bn.py --task cifar --opt sgd --mom 0.9 --wd 5e-4 --bs 256 --eval-recal 20"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
while ! grep -q ALLDONE $L/run_cifar0.log 2>/dev/null; do sleep 60; done
TM=$(/home/soso/advsci/mlip/bin/python -I -c "import json,sys; print([p for p in json.load(open(sys.argv[1]))['phases'] if p.get('matched_at')][0]['matched_at'])" $R/cifar_kpS_meta.json)
K=$C/cifar_kpS_end1.pt; F=$C/cifar_refS_s$TM.pt
set -x
[ -f $R/cifar_bn_own_meta.json ] || $PY --init $K --out cifar_bn_own --sched '[{"lr":0.04,"steps":2500}]' > $L/cifar_bn_own.log 2>&1
[ -f $R/cifar_bn_ref_meta.json ] || $PY --init $F --out cifar_bn_ref --sched '[{"lr":0.04,"steps":2500}]' > $L/cifar_bn_ref.log 2>&1
echo ALLDONE
