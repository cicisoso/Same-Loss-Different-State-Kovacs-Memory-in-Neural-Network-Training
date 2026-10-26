#!/bin/bash
# Replaces run_an3.sh, whose triggers (ALLDONE in run_k2.log and run_seeds.log) never come because those queues were taken
# over by the pool dispatchers. Same exploratory detail analyses (k_detail.py), same pairs and outputs, each run as soon as
# both runs of the pair have finished; pairs already analysed are skipped. Log: logs/kov/k_detail_all.log.
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
PAIRS="refL:kp_ta100 refL:kp_ta1000 refL:kp_h2 refL:kp_l2 refL:km2 refL_reset:kp_reset refL_r1:kp_r1 refL_r2:kp_r2"
for s in 1 2 3 4 5; do PAIRS="$PAIRS refL_sd$s:kp_sd$s refL_sd$s:km_sd$s"; done
while :; do
  left=0
  for p in $PAIRS; do
    b=${p%%:*}; k=${p##*:}
    { [ -f $R/k_detail_$k.json ] || [ -f $R/k_detail_$k.failed ]; } && continue
    left=1
    if [ -f $R/${b}_meta.json ] && [ -f $R/${k}_meta.json ]; then
      echo "$(date +%T) detail $k" >> $L/k_detail_all.log
      $P $C/k_detail.py $R $b $k $R/k_detail_$k.json >> $L/k_detail_all.log 2>&1
      [ -f $R/k_detail_$k.json ] || touch $R/k_detail_$k.failed
    fi
  done
  [ $left = 0 ] && break
  sleep 300
done
echo "$(date +%T) ALLDONE" >> $L/k_detail_all.log
