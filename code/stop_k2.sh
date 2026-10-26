#!/bin/bash
# Stop the serial withheld-protocol queue (run_k2.sh, tmux kov_k2) once refL_reset has finished: its remaining jobs (km2 and
# the two unseen-data replicates) run with identical commands as parallel chains of run_pool3.sh, which starts them only after
# the flag written here. The partial outputs of the km2 run interrupted by the stop are removed.
R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
pid_of() { ps -eo pid=,args= | awk -v t="--out $1 " '$2 ~ /python/ && index($0, t) {print $1}'; }
while [ ! -f $R/refL_reset_meta.json ]; do sleep 20; done
sleep 5
tmux kill-session -t kov_k2 2>/dev/null
sleep 5
for t in km2 refL_r1; do p=$(pid_of $t); [ -n "$p" ] && kill $p; done
sleep 5
for t in km2 refL_r1; do
  [ -f $R/${t}_meta.json ] || rm -f $R/${t}_eval.tsv $R/${t}_train.tsv $R/${t}_detail.npz $L/${t}.log
done
touch $L/k2_stopped
echo "$(date +%T) kov_k2 stopped after refL_reset; km2 and the replicates continue in run_pool3.sh" >> $L/run_k2.log
