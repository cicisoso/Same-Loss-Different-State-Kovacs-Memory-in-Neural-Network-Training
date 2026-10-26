#!/bin/bash
# E10 batch-norm thermostat test (CIFAR-10, momentum SGD). At the Kovacs match the heated run carries ~8% larger conv-filter
# norms, i.e. a ~15% lower effective learning rate of these scale-invariant filters, and its excursion is negative. A
# function-preserving rescaling of the filters to the other run's per-channel norms isolates the norm: four continuations on
# identical minibatches from the matched pair (Kovacs own / Kovacs at reference norms / reference own / reference at Kovacs norms).
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_small.py --task cifar --opt sgd --mom 0.9 --wd 5e-4 --bs 256"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
set -x
[ -f $C/cifar_kpS_end1.pt ] || $PY --init $C/cifar_base_s4000.pt --out cifar_kpS --match-every 1 --ref $R/cifar_refL_eval.tsv --sched '[{"lr":0.12,"steps":300},{"lr":0.005,"until":"match","max":3000,"save_end":1}]' > $L/cifar_kpS.log 2>&1
TM=$(/home/soso/advsci/mlip/bin/python -I -c "import json,sys; print([p for p in json.load(open(sys.argv[1]))['phases'] if p.get('matched_at')][0]['matched_at'])" $R/cifar_kpS_meta.json)
[ -f $C/cifar_refS_s$TM.pt ] || $PY --init $C/cifar_base_s4000.pt --out cifar_refS --sched "[{\"lr\":0.04,\"steps\":$((TM-4000))}]" --save-at $TM > $L/cifar_refS.log 2>&1
K=$C/cifar_kpS_end1.pt; F=$C/cifar_refS_s$TM.pt
[ -f $R/cifar_cn_own_meta.json ] || $PY --init $K --out cifar_cn_own --sched '[{"lr":0.04,"steps":2500}]' > $L/cifar_cn_own.log 2>&1
[ -f $R/cifar_cn_norm_meta.json ] || $PY --init $K --out cifar_cn_norm --sched "[{\"lr\":0.04,\"steps\":2500,\"rescale_from\":\"$F\"}]" > $L/cifar_cn_norm.log 2>&1
[ -f $R/cifar_cn_ref_meta.json ] || $PY --init $F --out cifar_cn_ref --sched '[{"lr":0.04,"steps":2500}]' > $L/cifar_cn_ref.log 2>&1
[ -f $R/cifar_cn_refnorm_meta.json ] || $PY --init $F --out cifar_cn_refnorm --sched "[{\"lr\":0.04,\"steps\":2500,\"rescale_from\":\"$K\"}]" > $L/cifar_cn_refnorm.log 2>&1
echo ALLDONE
