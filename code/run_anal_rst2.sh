#!/bin/bash
# Decision rule of PREREG note 10:36 on the optimizer resets (k_reset_rule.py), run when each reset analysis of
# run_anal_ann.sh lands (k_pairs_reset.json from the base checkpoint, k_pairs_rst.json from the transplant pair) and again
# when both exist. Output: results/kov/k_reset_rule.json; log: logs/kov/anal_reset_rule.log.
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
rule() { $P $C/k_reset_rule.py $R/k_reset_rule.json \
  "transplant|$R/k_pairs_rst.json|rs_kp:rs_ref|sw_kp_own:sw_ref_own" \
  "transplant repeat|$R/k_pairs_rst.json|rs_kp:rs_ref|nm_kp_sham:nm_ref_sham" \
  "base|$R/k_pairs_reset.json|kp_reset:refL_reset|kp:refL" >> $L/anal_reset_rule.log 2>&1; }
seen=""
while :; do
  for f in k_pairs_reset k_pairs_rst; do
    if [ -s $R/$f.json ] && [[ " $seen " != *" $f "* ]]; then sleep 5; echo "$(date +%T) $f" >> $L/anal_reset_rule.log; rule; seen="$seen $f"; fi
  done
  [[ "$seen" == *k_pairs_reset*k_pairs_rst* || "$seen" == *k_pairs_rst*k_pairs_reset* ]] && break
  sleep 60
done
echo "$(date +%T) ALLDONE" >> $L/anal_reset_rule.log
