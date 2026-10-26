#!/bin/bash
# Final linear-regime analyses (PREREG E10: +-20% steps and pulses, two-pulse superposition), after the last run
# (pulse_cold_late) finished: k_lin.py on every constant-step run (written to k_lin_final.json; k_lin.json stays the
# fit-set input of figs/fig4_data.py), the SI summary k_lin_report.py, and the frozen registered models (fit set as in
# run_an2b.sh) on the withheld linear-regime runs, scored by whole-run absolute-loss RMSE. Log: logs/kov/linfinal.log.
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
FIT=pulse_hot,pulse_cold,step_1.25e-4,step_3e-3,step_2.5e-4,step_2e-3,step_5e-4
LIN=step_1.2e-3,step_8e-4,pulse_1.2e-3,pulse_8e-4,pulse_two,pulse_cold_late
echo "$(date +%T) start"
$P $C/k_lin.py $R ref step_1.25e-4,step_2.5e-4,step_5e-4,step_8e-4,step_1.2e-3,step_2e-3,step_3e-3 $R/k_lin_final.json > $L/k_lin_final.log 2>&1
echo "$(date +%T) k_lin ($?)"
$P $C/k_lin_report.py $R ref > $L/k_lin_report.log 2>&1
echo "$(date +%T) k_lin_report ($?)"
$P $C/k_models.py $R ref $FIT $LIN $R/k_models_lin.json > $L/k_models_lin.log 2>&1
echo "$(date +%T) k_models_lin ($?)"
echo "$(date +%T) ALLDONE"
