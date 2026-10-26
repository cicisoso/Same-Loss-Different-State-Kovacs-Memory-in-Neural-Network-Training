#!/bin/bash
# Render every figure of the manuscript and the Supporting Information under its final name into the paper directory,
# with the collision audit of each exported PDF. Data extraction runs first (fig*_data.py); missing inputs leave a panel
# empty and are reported by the plotting script. Usage: bash render_all.sh [out_dir=/home/soso/advsci/paper/figs]
P=/home/soso/advsci/mlip/bin/python; R=/home/soso/advsci/results/kov; F=/home/soso/advsci/code/kov/figs
O=${1:-/home/soso/advsci/paper/figs}; D=$O/data; mkdir -p $O $D
cd $F || exit 1
SEEDS=refL_sd1:kp_sd1,refL_sd2:kp_sd2,refL_sd3:kp_sd3,refL_sd4:kp_sd4,refL_sd5:kp_sd5,ref:kp
csv() { local f out=""; for f in "$@"; do [ -f "$f" ] && out="$out${out:+,}$f"; done; echo "$out"; }
PLAW=$(csv $R/k_peaklaw_expl_gpt.json $R/k_peaklaw_expl_small.json $R/k_peaklaw_interim2.json $R/k_peaklaw_withheld.json $R/k_peaklaw_seeds.json $R/k_peaklaw_extra.json $R/k_peaklaw_status.json)
CONT=$(csv $R/k_models_full.json $R/k_models_withheld.json $R/k_models_seeds.json $R/k_models_extra.json)
DET=$(csv $R/k_detail_kp_sd1.json $R/k_detail_kp_sd2.json $R/k_detail_kp_sd3.json $R/k_detail_kp_sd4.json $R/k_detail_kp_sd5.json)
[ -n "$DET" ] || DET=$R/k_detail_kp_ta100.json
$P fig1_data.py $R $D/fig1_data.json $SEEDS 2>&1 | tail -1
$P fig1_phenom.py $D/fig1_data.json $O/fig1 2>&1 | tail -1
$P fig3_data.py $R $D/fig3_data.json 2>&1 | tail -1
$P fig3_mech.py $D/fig3_data.json $O/fig2 2>&1 | tail -1                 # manuscript Figure 2: state interventions
$P fig2_theory.py $R $O/fig3 2>&1 | tail -1                              # manuscript Figure 3: restricted model
$P fig4_data.py $D/fig4_data.json $R/k_super_kp.json $CONT $PLAW 2>&1 | tail -1
$P fig4_pred.py $D/fig4_data.json $O/fig4 2>&1 | tail -1
$P fig5_data.py $R $D/fig5_data.json 2>&1 | tail -1
$P fig5_breadth.py $D/fig5_data.json $O/fig5 2>&1 | tail -1
$P fig6_data.py $R $D/fig6_data.json 2>&1 | tail -1
$P fig6_conseq.py $D/fig6_data.json $O/fig6 2>&1 | tail -1
$P figS1_cifar.py $R $O/figS1 2>&1 | tail -1
$P -I /home/soso/advsci/code/kov/k_anneal_fbin.py $R B0=an_kp_B0:an_ref_B0,B200=an_kp_B200:an_ref_B200,B1000=an_kp_B1000:an_ref_B1000,AN1=anM_kp_sd1:anM_ref_sd1,AN2=anM_kp_sd2:anM_ref_sd2,AN3=anM_kp_sd3:anM_ref_sd3 $R/k_anneal_fbin.json > $D/k_anneal_fbin.log 2>&1
$P figS2_tokens.py $DET $O/figS2 $R/k_anneal_fbin.json 2>&1 | tail -1
$P toc_image.py $D/fig1_data.json $O/toc 2>&1 | tail -1
for f in fig1 fig2 fig3 fig4 fig5 fig6 figS1 figS2 toc; do
  [ -f $O/$f.pdf ] || { echo "$f: missing"; continue; }
  echo "$f: $($P audit_figure_collisions.py $O/$f.pdf --json-out $D/$f.collision-audit.json 2>&1 | grep -E '^verdict|^summary' | tr '\n' ' ')"
done
