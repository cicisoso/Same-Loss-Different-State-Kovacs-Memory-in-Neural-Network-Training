#!/bin/bash
# Rerun the exact-model checks after correcting the batch-noise coefficient (1 + 2/B -> 1 + 1/B, Monte Carlo verified).
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov
set -x
$P $C/ts_kappa.py $R/ts_kappa.json &
$P $C/ts_noise.py $R/ts_noise.json &
$P $C/ts_peak.py $R/ts_peak.json &
$P $C/ts_sectors.py $R/ts_sectors.json &
$P $C/ts_exact.py $R/ts_exact_a1.5.json 1.5 &
wait
echo ALLDONE
