#!/bin/bash
# Stage-1 analyses: blind contest on the GPT fit set (pre-registered fit tags), scoring, linear response of the five steps.
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
while ! grep -q ALLDONE $L/run_k1.log 2>/dev/null; do sleep 30; done
set -x
FIT=pulse_hot,pulse_cold,step_1.25e-4,step_3e-3,step_2.5e-4,step_2e-3,step_5e-4
$P $C/k_lin.py $R ref step_1.25e-4,step_2.5e-4,step_5e-4,step_2e-3,step_3e-3 $R/k_lin.json > $L/k_lin.log 2>&1
$P $C/k_models.py $R ref $FIT kp $R/k_models_full.json > $L/k_models_full.log 2>&1
$P $C/k_score.py $R/k_models_full.json > $L/k_score_full.log 2>&1
echo ALLDONE
