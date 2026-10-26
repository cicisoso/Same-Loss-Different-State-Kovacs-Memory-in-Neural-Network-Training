#!/bin/bash
# Stage-2 analyses as registered in PREREG.md: the blind contest and the peak-time law on the withheld protocols (after
# run_k2.sh), then on the independent seeds (after run_seeds.sh). Models are fitted on the registered fit set only.
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
FIT=pulse_hot,pulse_cold,step_1.25e-4,step_3e-3,step_2.5e-4,step_2e-3,step_5e-4
W=kp_ta100:refL,kp_ta1000:refL,kp_h2:refL,kp_l2:refL,km2:refL,kp_r1:refL_r1,kp_r2:refL_r2
PL=kp_ta100:refL,kp_ta1000:refL,kp_h2:refL,kp_l2:refL,kp_r1:refL_r1,kp_r2:refL_r2
SD=kp_sd1:refL_sd1,kp_sd2:refL_sd2,kp_sd3:refL_sd3,kp_sd4:refL_sd4,kp_sd5:refL_sd5
while ! grep -q ALLDONE $L/run_k2.log 2>/dev/null; do sleep 60; done
set -x
$P $C/k_models.py $R ref $FIT $W $R/k_models_withheld.json > $L/k_models_withheld.log 2>&1
$P $C/k_score.py $R/k_models_withheld.json > $L/k_score_withheld.log 2>&1
$P $C/k_peaklaw.py $R $PL 0 1 1 $R/k_peaklaw_withheld.json > $L/k_peaklaw_withheld.log 2>&1
$P $C/k_pairs.py $R km2:refL,kp_reset:refL_reset $R/k_pairs_k2.json > $L/k_pairs_k2.log 2>&1
echo K2DONE
while ! grep -q ALLDONE $L/run_seeds.log 2>/dev/null; do sleep 60; done
$P $C/k_models.py $R ref $FIT $SD $R/k_models_seeds.json > $L/k_models_seeds.log 2>&1
$P $C/k_score.py $R/k_models_seeds.json > $L/k_score_seeds.log 2>&1
$P $C/k_peaklaw.py $R $SD 0 1 1 $R/k_peaklaw_seeds.json > $L/k_peaklaw_seeds.log 2>&1
echo ALLDONE
