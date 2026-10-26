#!/bin/bash
# Replaces run_an2.sh, whose waits for ALLDONE in run_k2.log and run_seeds.log can no longer end (those chains moved to the
# pool dispatchers). The same stage-2 contest as registered in PREREG.md (k_models.py fitted on the registered fit set
# only, then k_score.py) on the withheld pairs and on the seeds whose two runs have finished, rerun whenever another pair
# finishes. The peak-time law is scored per run by k_peaklaw_status.py (make_tables.sh). The reverse protocol km2 gets its
# pair statistics from its release. Ends when every pair is in, or when logs/kov/an2b.stop exists.
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
FIT=pulse_hot,pulse_cold,step_1.25e-4,step_3e-3,step_2.5e-4,step_2e-3,step_5e-4
W="kp_ta100:refL kp_ta1000:refL kp_h2:refL kp_l2:refL km2:refL kp_r1:refL_r1 kp_r2:refL_r2"
SD="kp_sd1:refL_sd1 kp_sd2:refL_sd2 kp_sd3:refL_sd3 kp_sd4:refL_sd4 kp_sd5:refL_sd5"
TMF() { /home/soso/advsci/mlip/bin/python -I -c "import json,sys; print([p for p in json.load(open(sys.argv[1]))['phases'] if p.get('matched_at')][0]['matched_at'])" $1; }
fin() { local o="" p; for p in $1; do [ -s $R/${p%%:*}_meta.json ] && [ -s $R/${p##*:}_meta.json ] && o="$o,$p"; done; echo ${o#,}; }
lw=""; lsd=""; lk=""
while [ ! -f $L/an2b.stop ]; do
  w=$(fin "$W"); s=$(fin "$SD")
  if [ -n "$w" ] && [ "$w" != "$lw" ]; then
    echo "$(date +%T) withheld: $w"
    $P $C/k_models.py $R ref $FIT $w $R/k_models_withheld.json > $L/k_models_withheld.log 2>&1 &&
      $P $C/k_score.py $R/k_models_withheld.json > $L/k_score_withheld.log 2>&1
    echo "$(date +%T) withheld done ($?)"; lw=$w
  fi
  if [ -n "$s" ] && [ "$s" != "$lsd" ]; then
    echo "$(date +%T) seeds: $s"
    $P $C/k_models.py $R ref $FIT $s $R/k_models_seeds.json > $L/k_models_seeds.log 2>&1 &&
      $P $C/k_score.py $R/k_models_seeds.json > $L/k_score_seeds.log 2>&1
    echo "$(date +%T) seeds done ($?)"; lsd=$s
  fi
  if [ -z "$lk" ] && [ -s $R/km2_meta.json ] && [ -s $R/refL_meta.json ]; then
    T=$(TMF $R/km2_meta.json) && $P $C/k_pairs.py $R km2:refL@$T $R/k_pairs_k2.json > $L/k_pairs_k2.log 2>&1
    echo "$(date +%T) km2 pair statistics ($?)"; lk=1
  fi
  [ "$w" = "$(echo $W | tr ' ' ,)" ] && [ "$s" = "$(echo $SD | tr ' ' ,)" ] && [ -n "$lk" ] && break
  sleep 300
done
echo ALLDONE
