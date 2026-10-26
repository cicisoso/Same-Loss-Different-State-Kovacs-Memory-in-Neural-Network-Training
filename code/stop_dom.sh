#!/bin/bash
# Stop the domain queue (run_dom.sh, tmux kov_dom) once pubmed_km has finished: the PubMed pulses and constant steps that
# follow it are used by no registered analysis or figure, and their GPU slot goes to the seed chains of the pool.
# Partial outputs of the run interrupted by the stop are removed.
R=/home/soso/advsci/results/kov; L=/home/soso/advsci/logs/kov
while [ ! -f $R/pubmed_km_meta.json ]; do sleep 20; done
sleep 5
tmux kill-session -t kov_dom 2>/dev/null
sleep 5
for t in pubmed_pulse_hot pubmed_pulse_cold pubmed_step_1.25e-4 pubmed_step_3e-3; do
  [ -f $R/${t}_meta.json ] || rm -f $R/${t}_eval.tsv $R/${t}_train.tsv $R/${t}_detail.npz $L/${t}.log
done
echo "$(date +%T) kov_dom stopped after pubmed_km" >> $L/run_dom.log
