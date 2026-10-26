#!/bin/bash
# Rebuild every Supporting-Information table from the analysis outputs (no training): Table S2 (k_table.py), S3 (k_interv_table.py
# with both reset pairs), S4 (k_anneal_table.py), S6 (k_anneal_rep_table.py), S5 (k_peaklaw_status.py + k_peaklaw_table.py),
# S7 (k_contest_table.py). Missing inputs give "pending" rows; MISSING="not run" labels them for the final tables.
# Usage: [MISSING=label] bash make_tables.sh [out_dir=/home/soso/advsci/paper/tables]
P="/home/soso/advsci/mlip/bin/python -I"; C=/home/soso/advsci/code/kov; R=/home/soso/advsci/results/kov
T=${1:-/home/soso/advsci/paper/tables}; mkdir -p $T
$P $C/k_table.py $R $R/k_table.json $T/k_table.tex "${MISSING:-pending}" > /dev/null && echo "S2 ok"
$P $C/k_interv_table.py $R/k_norm.json $R/k_swap.json $T/k_interv.tex $R/k_pairs_rst.json $R/k_pairs_reset.json > /dev/null && echo "S3 ok"
MOD=$R/k_models_anneal.json,$R/k_models_anneal_sd.json
$P $C/k_anneal_table.py $T/k_anneal.tex $MOD "Transplant|$R/k_anneal_pairs_tp.json|B0sw" "|$R/k_anneal_pairs_tp.json|B200sw" \
  "|$R/k_anneal_pairs_tp.json|B1000sw" "Kovacs rescaled|$R/k_anneal_pairs_N.json|GN" > /dev/null && echo "S4 ok"   # seeds: S6
$P $C/k_anneal_rep_table.py $R $R/k_anneal_pairs_sd.json $T/k_anneal_rep.tex $R/k_anneal_rep.json > /dev/null && echo "S6 ok"
$P $C/k_peaklaw_status.py $R $R/k_peaklaw_status.json | head -1 && $P $C/k_peaklaw_table.py $R/k_peaklaw_status.json $T/k_peaklaw.tex > /dev/null && echo "S5 ok"
$P $C/k_contest_table.py $T/k_contest.tex $R/k_contest.json exploratory=$R/k_models_full.json withheld=$R/k_models_withheld.json \
  seeds=$R/k_models_seeds.json extra=$R/k_models_extra.json > /dev/null && echo "S7 ok"
