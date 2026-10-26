#!/bin/bash
# Stage-2 contest (PREREG.md, "Blind prediction contest") on the registered Kovacs cases that run_an2b.sh does not cover:
# the cool->heat runs of the five seeds (the seed Kovacs runs are run in both directions) and the ageing pair (kpA8
# against refA8, protocol from the reference checkpoint at step 8000). Same code as run_an2b.sh: k_models.py fitted on
# the registered fit set only, then k_score.py, rerun whenever another pair finishes. Ends when every pair is in, or when
# logs/kov/an2c.stop exists. Output: results/kov/k_models_extra.json (+ _score.json).
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
FIT=pulse_hot,pulse_cold,step_1.25e-4,step_3e-3,step_2.5e-4,step_2e-3,step_5e-4
X="km_sd1:refL_sd1 km_sd2:refL_sd2 km_sd3:refL_sd3 km_sd4:refL_sd4 km_sd5:refL_sd5 kpA8:refA8"
fin() { local o="" p; for p in $1; do [ -s $R/${p%%:*}_meta.json ] && [ -s $R/${p##*:}_meta.json ] && o="$o,$p"; done; echo ${o#,}; }
lx=""
while [ ! -f $L/an2c.stop ]; do
  x=$(fin "$X")
  if [ -n "$x" ] && [ "$x" != "$lx" ]; then
    echo "$(date +%T) extra: $x"
    $P $C/k_models.py $R ref $FIT $x $R/k_models_extra.json > $L/k_models_extra.log 2>&1 &&
      $P $C/k_score.py $R/k_models_extra.json > $L/k_score_extra.log 2>&1
    echo "$(date +%T) extra done ($?)"; lx=$x
  fi
  [ "$x" = "$(echo $X | tr " " ,)" ] && break
  sleep 300
done
echo ALLDONE
