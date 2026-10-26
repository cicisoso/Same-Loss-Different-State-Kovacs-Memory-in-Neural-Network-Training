#!/bin/bash
# Exploratory detail analyses (k_detail.py: sequence bootstrap, token-frequency and position decomposition) of every Kovacs
# pair with detail logs, run as each queue finishes: withheld family (after run_k2.sh), then seeds (after run_seeds.sh).
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
det() { [ -f $R/k_detail_$2.json ] || $P $C/k_detail.py $R $1 $2 $R/k_detail_$2.json >> $L/k_detail_all.log 2>&1; }
while ! grep -q ALLDONE $L/run_k2.log 2>/dev/null; do sleep 60; done
set -x
for k in kp_ta100 kp_ta1000 kp_h2 kp_l2 km2; do det refL $k; done
det refL_reset kp_reset; det refL_r1 kp_r1; det refL_r2 kp_r2
echo K2DONE
while ! grep -q ALLDONE $L/run_seeds.log 2>/dev/null; do sleep 60; done
for s in 1 2 3 4 5; do det refL_sd$s kp_sd$s; det refL_sd$s km_sd$s; done
echo ALLDONE
