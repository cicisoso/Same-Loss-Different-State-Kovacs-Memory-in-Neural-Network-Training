#!/bin/bash
# E10 exploratory (PREREG addendum 2026-10-20 09:44): identical anneals (B = 0; linear decay to 0 over 500 steps, the
# schedule of run_swap.sh) from the matched checkpoints after the function-preserving norm changes of the norm test:
# the Kovacs network with both reductions (as nm_kp_both) and the reference with both increases to the Kovacs norms.
# Starts when run_swap.sh has finished its last anneal, so that it takes that GPU slot.
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
while [ ! -f $R/an_ref_B1000_meta.json ]; do sleep 20; done
K=$C/kpS_end1.pt; F=$C/refS_s4987.pt
S='[{"lr":1e-3,"steps":500,"decay":1}]'; E="--dense 10 --sparse 10"
set -x
run() {   # run OUT ARGS...: skip if done; after a CUDA out-of-memory failure wait 5 min and retry (three tries)
  local out=$1 try; shift
  for try in 1 2 3; do
    [ -f $R/${out}_meta.json ] && return 0
    $PY "$@" --out $out --sched "$S" $E > $L/$out.log 2>&1
    grep -q "out of memory" $L/$out.log || return 0
    sleep 300
  done
}
run an_kpN_B0 --init $K --rescale-resid to:$F --rescale-proj to:$F
run an_refN_B0 --init $F --rescale-resid to:$K --rescale-proj to:$K
echo ALLDONE
