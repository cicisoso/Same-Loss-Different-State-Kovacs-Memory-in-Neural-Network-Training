#!/bin/bash
# Exploratory diagnostics on the stage-1 contest (kp is exploratory): per-run fit errors and the p = q = 1 spectral model.
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
FIT=pulse_hot,pulse_cold,step_1.25e-4,step_3e-3,step_2.5e-4,step_2e-3,step_5e-4
$P $C/k_models_x.py $R ref $FIT kp $R/k_models_diag.json > $L/k_models_diag.log 2>&1
$P $C/k_score.py $R/k_models_diag.json > $L/k_score_diag.log 2>&1
echo ALLDONE
