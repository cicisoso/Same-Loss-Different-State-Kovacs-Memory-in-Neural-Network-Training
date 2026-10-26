#!/bin/bash
# Analyses (no training) of the anneals and resets as their runs finish, each group waiting for its own inputs in the
# background: B = 1000 with the anneal model contest (k_anneal.py, k_anneal_pairs.py, k_models.py), the norm-changed
# anneals (k_annealN.py), the reset pairs (k_pairs.py from the match, PREREG note 10:36), and the anneal replications on
# seeds 1-3 (k_anneal_pairs.py, k_models.py), re-run after each seed. Logs in logs/kov/anal_ann_*.log.
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
FIT=pulse_hot,pulse_cold,step_1.25e-4,step_3e-3,step_2.5e-4,step_2e-3,step_5e-4
TMF() { /home/soso/advsci/mlip/bin/python -I -c "import json,sys; print([p for p in json.load(open(sys.argv[1]))['phases'] if p.get('matched_at')][0]['matched_at'])" $1; }
waitf() { local f; for f in "$@"; do while [ ! -f $f ]; do sleep 30; done; done; }
TP="B0sw=an_kp_B0:an_ref_B0:sw_kp_own:sw_ref_own:0,B0nm=an_kp_B0:an_ref_B0:nm_kp_sham:nm_ref_sham:0"
TP="$TP,B200sw=an_kp_B200:an_ref_B200:sw_kp_own:sw_ref_own:200,B200nm=an_kp_B200:an_ref_B200:nm_kp_sham:nm_ref_sham:200"
TP="$TP,B1000sw=an_kp_B1000:an_ref_B1000:sw_kp_own:sw_ref_own:1000,B1000nm=an_kp_B1000:an_ref_B1000:nm_kp_sham:nm_ref_sham:1000"
SD="AN1=anM_kp_sd1:anM_ref_sd1:kp_sd1:refL_sd1:0,AN2=anM_kp_sd2:anM_ref_sd2:kp_sd2:refL_sd2:0,AN3=anM_kp_sd3:anM_ref_sd3:kp_sd3:refL_sd3:0"

( waitf $R/an_ref_B1000_meta.json
  echo "$(date +%T) B1000"
  $P $C/k_anneal.py $R 0,200,1000 > $L/anal_ann_k_anneal.log 2>&1
  $P $C/k_anneal_pairs.py $R "$TP" $R/k_anneal_pairs_tp.json > $L/anal_ann_pairs_tp.log 2>&1
  $P $C/k_models.py $R ref $FIT an_kp_B0:an_ref_B0,an_kp_B200:an_ref_B200,an_kp_B1000:an_ref_B1000 $R/k_models_anneal.json > $L/anal_ann_models.log 2>&1
  echo "$(date +%T) B1000 done" ) &

( waitf $R/an_kpN_B0_meta.json $R/an_refN_B0_meta.json
  echo "$(date +%T) annN"
  $P $C/k_annealN.py $R > $L/anal_ann_annN.log 2>&1
  echo "$(date +%T) annN done" ) &

( waitf $R/rs_kp_meta.json $R/rs_ref_meta.json
  echo "$(date +%T) RST"
  $P $C/k_pairs.py $R rs_kp:rs_ref@4987,sw_kp_own:sw_ref_own@4987,nm_kp_sham:nm_ref_sham@4987 $R/k_pairs_rst.json > $L/anal_ann_rst.log 2>&1
  echo "$(date +%T) RST done" ) &

( waitf $R/kp_reset_meta.json $R/refL_reset_meta.json
  echo "$(date +%T) reset from base"
  T1=$(TMF $R/kp_reset_meta.json); T0=$(TMF $R/kp_meta.json)
  $P $C/k_pairs.py $R kp_reset:refL_reset@$T1,kp:refL@$T0 $R/k_pairs_reset.json > $L/anal_ann_reset.log 2>&1
  echo "$(date +%T) reset from base done" ) &

( for s in 1 2 3; do
    waitf $R/anM_kp_sd${s}_meta.json $R/anM_ref_sd${s}_meta.json $R/kp_sd${s}_meta.json
    echo "$(date +%T) AN$s"
    $P $C/k_anneal_pairs.py $R "$SD" $R/k_anneal_pairs_sd.json > $L/anal_ann_pairs_sd.log 2>&1
  done
  $P $C/k_models.py $R ref $FIT anM_kp_sd1:anM_ref_sd1,anM_kp_sd2:anM_ref_sd2,anM_kp_sd3:anM_ref_sd3 $R/k_models_anneal_sd.json > $L/anal_ann_models_sd.log 2>&1
  echo "$(date +%T) AN done" ) &
wait
echo ALLDONE
