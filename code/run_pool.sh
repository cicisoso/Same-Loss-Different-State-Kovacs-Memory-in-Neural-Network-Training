#!/bin/bash
# Pool for the remaining 30M and 124M runs. With six 30M runs on the GPU (7.5 GB each) the queues that were still waiting would
# have pushed it past 48 GB, and the seed queue ran its 17 remaining runs one after another. Here the same jobs (identical
# commands, log names and outputs as run_seeds.sh, run_lin.sh, run_k3.sh, run_wd0.sh and run_big.sh) are grouped into chains
# that run in order; a chain starts, in the priority order below, only when the GPU has room for it on three readings spread
# over four minutes. run_seeds.sh is stopped at the boundary after kp_sd1, and its ALLDONE line is written here once every seed
# has both protocols, so run_an2.sh and run_an3.sh continue unchanged.
PY="/home/soso/advsci/mlip/bin/python -I /home/soso/advsci/code/kov/kov_train.py --eval-seqs 512"
PYW="$PY --wd 0"; PYB="$PY --layers 12 --dim 768 --heads 12"
R=/home/soso/advsci/results/kov; C=/home/soso/advsci/ckpt/kov; L=/home/soso/advsci/logs/kov
I0="--init $C/base_s4000.pt"
KP() { echo "[{\"lr\":$1,\"steps\":$2},{\"lr\":$3,\"until\":\"match\",\"max\":3000},{\"lr\":1e-3,\"steps\":2500}]"; }
TMF() { /home/soso/advsci/mlip/bin/python -I -c "import json,sys; print([p for p in json.load(open(sys.argv[1]))['phases'] if p.get('matched_at')][0]['matched_at'])" $1; }
BASE='[{"lr":1e-3,"steps":500,"warmup":1},{"lr":1e-3,"steps":3500}]'; REF='[{"lr":1e-3,"steps":4500}]'
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_room() { local i; for i in 1 2 3; do [ "$(gpu_free)" -ge "$1" ] || return 1; [ $i -lt 3 ] && sleep 120; done; return 0; }
need() { [ $1 = BIG ] && echo 21000 || echo 9100; }               # MiB: 7.5 GB per 30M run plus margin; 124M estimated
pid_of() { ps -eo pid=,args= | awk -v t="--out $1 " '$2 ~ /python/ && index($0, t) {print $1}'; }

job() {   # job DONE LOG CMD...: skip if DONE exists; after a CUDA out-of-memory failure wait 10 min and retry (three tries)
  local done=$1 log=$2 try; shift 2
  for try in 1 2 3; do
    [ -f "$done" ] && return 0
    echo "$(date +%T) start $(basename $log .log) (try $try)"
    "$@" > "$log" 2>&1
    [ -f "$done" ] && { echo "$(date +%T) done $(basename $log .log)"; return 0; }
    grep -q "out of memory" "$log" || { echo "$(date +%T) FAILED $(basename $log .log)"; return 1; }
    echo "$(date +%T) OOM $(basename $log .log)"; sleep 600
  done
  return 1
}

seed_chain() {   # base, reference, heat->cool and cool->heat for one seed (commands of run_seeds.sh)
  local sd=$1 A="--seed $1 --data-seed $1" I="--init $C/base_sd$1_s4000.pt"
  job $C/base_sd${sd}_s4000.pt $L/base_sd$sd.log $PY $A --out base_sd$sd --sched "$BASE" --save-at 4000 || return 1
  job $R/refL_sd${sd}_meta.json $L/refL_sd$sd.log $PY $A $I --out refL_sd$sd --sched "$REF" --dense 10 --sparse 10 || return 1
  job $R/kp_sd${sd}_meta.json $L/kp_sd$sd.log $PY $A $I --out kp_sd$sd --match-every 2 --ref $R/refL_sd${sd}_eval.tsv --sched "$(KP 3e-3 300 1.25e-4)" || return 1
  job $R/km_sd${sd}_meta.json $L/km_sd$sd.log $PY $A $I --out km_sd$sd --match-every 1 --ref $R/refL_sd${sd}_eval.tsv --sched "$(KP 1.25e-4 300 1.5e-3)"
}
chain_AGE() {    # the heat->cool protocol from the reference checkpoint at step 8000 (run_lin.sh)
  local I8="--init $C/refL_s8000.pt"
  job $R/refA8_meta.json $L/refA8.log $PY $I8 --out refA8 --sched "$REF" --dense 10 --sparse 10 || return 1
  job $R/kpA8_meta.json $L/kpA8.log $PY $I8 --out kpA8 --match-every 1 --ref $R/refA8_eval.tsv --sched '[{"lr":3e-3,"steps":300},{"lr":1.25e-4,"until":"match","max":3000},{"lr":1e-3,"steps":2500}]'
}
probe_chain() {  # quench probes at the match and the reference quenched at the same step (run_k3.sh)
  local k=$1 A B TM
  if [ $k = kp ]; then A=3e-3; B=1.25e-4; else A=1.25e-4; B=1.5e-3; fi
  job $R/${k}_probe_meta.json $L/${k}_probe.log $PY $I0 --out ${k}_probe --match-every 1 --ref $R/refL_eval.tsv --sched "[{\"lr\":$A,\"steps\":300},{\"lr\":$B,\"until\":\"match\",\"max\":3000},{\"lr\":1.25e-4,\"steps\":400}]" || return 1
  TM=$(TMF $R/${k}_probe_meta.json)
  job $R/ref_probe_${k}_meta.json $L/ref_probe_$k.log $PY $I0 --out ref_probe_$k --sched "[{\"lr\":1e-3,\"steps\":$((TM-4000))},{\"lr\":1.25e-4,\"steps\":400}]"
}
chain_WD0() {    # zero weight decay (run_wd0.sh)
  local I="--init $C/base_wd0_s4000.pt"
  job $C/base_wd0_s4000.pt $L/base_wd0.log $PYW --out base_wd0 --sched "$BASE" --save-at 4000 || return 1
  job $R/refL_wd0_meta.json $L/refL_wd0.log $PYW $I --out refL_wd0 --sched "$REF" --dense 10 --sparse 10 || return 1
  job $R/kp_wd0_meta.json $L/kp_wd0.log $PYW $I --out kp_wd0 --ref $R/refL_wd0_eval.tsv --sched "$(KP 3e-3 300 1.25e-4)" || return 1
  job $R/km_wd0_meta.json $L/km_wd0.log $PYW $I --out km_wd0 --match-every 1 --ref $R/refL_wd0_eval.tsv --sched "$(KP 1.25e-4 300 1.5e-3)" || return 1
  job $R/pulse_hot_wd0_meta.json $L/pulse_hot_wd0.log $PYW $I --out pulse_hot_wd0 --sched '[{"lr":3e-3,"steps":300},{"lr":1e-3,"steps":2700}]' || return 1
  job $R/pulse_cold_wd0_meta.json $L/pulse_cold_wd0.log $PYW $I --out pulse_cold_wd0 --sched '[{"lr":1.25e-4,"steps":300},{"lr":1e-3,"steps":2700}]'
}
lin_chain() {    # small step and pulse at one learning rate, then one superposition test (run_lin.sh)
  local lr=$1 last=$2
  job $R/step_${lr}_meta.json $L/step_$lr.log $PY $I0 --out step_$lr --sched "[{\"lr\":$lr,\"steps\":3000}]" || return 1
  job $R/pulse_${lr}_meta.json $L/pulse_$lr.log $PY $I0 --out pulse_$lr --sched "[{\"lr\":$lr,\"steps\":300},{\"lr\":1e-3,\"steps\":2700}]" || return 1
  if [ $last = pulse_two ]; then
    job $R/pulse_two_meta.json $L/pulse_two.log $PY $I0 --out pulse_two --sched '[{"lr":3e-3,"steps":300},{"lr":1e-3,"steps":300},{"lr":1.25e-4,"steps":300},{"lr":1e-3,"steps":2100}]'
  else
    job $R/pulse_cold_late_meta.json $L/pulse_cold_late.log $PY $I0 --out pulse_cold_late --sched '[{"lr":1e-3,"steps":600},{"lr":1.25e-4,"steps":300},{"lr":1e-3,"steps":2100}]'
  fi
}
chain_BIG() {    # GPT-124M (run_big.sh)
  local I="--init $C/big_base_s4000.pt"
  job $C/big_base_s4000.pt $L/big_base.log $PYB --out big_base --sched "$BASE" --save-at 4000 || return 1
  job $R/big_refL_meta.json $L/big_refL.log $PYB $I --out big_refL --sched "$REF" --dense 10 --sparse 10 || return 1
  job $R/big_kp_meta.json $L/big_kp.log $PYB $I --out big_kp --match-every 2 --ref $R/big_refL_eval.tsv --sched "$(KP 3e-3 300 1.25e-4)" || return 1
  job $R/big_pulse_hot_meta.json $L/big_pulse_hot.log $PYB $I --out big_pulse_hot --sched '[{"lr":3e-3,"steps":300},{"lr":1e-3,"steps":2700}]' || return 1
  job $R/big_pulse_cold_meta.json $L/big_pulse_cold.log $PYB $I --out big_pulse_cold --sched '[{"lr":1.25e-4,"steps":300},{"lr":1e-3,"steps":2700}]'
}
run_chain() {
  case $1 in
    S1) local A="--seed 1 --data-seed 1" I="--init $C/base_sd1_s4000.pt"
        job $R/kp_sd1_meta.json $L/kp_sd1.log $PY $A $I --out kp_sd1 --match-every 2 --ref $R/refL_sd1_eval.tsv --sched "$(KP 3e-3 300 1.25e-4)" &&
        job $R/km_sd1_meta.json $L/km_sd1.log $PY $A $I --out km_sd1 --match-every 1 --ref $R/refL_sd1_eval.tsv --sched "$(KP 1.25e-4 300 1.5e-3)";;
    S[2-5]) seed_chain ${1#S};;
    AGE) chain_AGE;; PRkp) probe_chain kp;; PRkm) probe_chain km;; WD0) chain_WD0;;
    LINa) lin_chain 1.2e-3 pulse_two;; LINb) lin_chain 8e-4 pulse_cold_late;; BIG) chain_BIG;;
  esac
  echo "$(date +%T) chain $1 exit $?"
}

stop_seeds() {   # stop run_seeds.sh as soon as kp_sd1 has ended, before its next run (base_sd2) gets going
  while [ -n "$(pid_of kp_sd1)" ]; do sleep 2; done
  tmux kill-session -t kov_seeds 2>/dev/null; sleep 3
  for t in base_sd2 km_sd1; do p=$(pid_of $t); [ -n "$p" ] && kill $p; done
  sleep 5; touch $L/seeds_stopped; echo "$(date +%T) run_seeds.sh stopped after kp_sd1 (meta: $(ls $R/kp_sd1_meta.json 2>/dev/null))"
}
mark_seeds() {   # ALLDONE for run_seeds.sh once every seed has both protocols
  local sd ok
  while :; do
    ok=1; for sd in 1 2 3 4 5; do [ -f $R/kp_sd${sd}_meta.json ] && [ -f $R/km_sd${sd}_meta.json ] || ok=0; done
    [ $ok = 1 ] && break; sleep 60
  done
  echo "+ seeds 2-5 and km_sd1 were run by run_pool.sh"; echo ALLDONE
}

stop_seeds &
mark_seeds >> $L/run_seeds.log 2>&1 &
todo="S2 S3 S4 S5 S1 AGE PRkp PRkm WD0 LINa LINb BIG"
first_ready() { local c; for c in $todo; do case $c in S*) [ -f $L/seeds_stopped ] || continue;; esac; echo $c; return; done; }
while [ -n "$todo" ]; do
  c=$(first_ready)
  if [ -n "$c" ] && gpu_room $(need $c); then
    echo "$(date +%T) pool: start chain $c (free $(gpu_free) MiB)"
    run_chain $c >> $L/pool_$c.log 2>&1 &
    todo=$(echo " $todo " | sed "s/ $c / /; s/^ *//; s/ *\$//")
    sleep 240
  else
    sleep 60
  fi
done
wait
echo ALLDONE
