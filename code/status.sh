#!/bin/bash
# One line per running kov_train job: output tag, last logged step, elapsed time.
R=/home/soso/advsci/results/kov
for pid in $(pgrep -f "kov/kov_train.py"); do
  out=$(tr "\0" " " < /proc/$pid/cmdline | grep -o -- "--out [^ ]*" | cut -d" " -f2)
  st=$(tail -1 $R/${out}_train.tsv 2>/dev/null | cut -f1)
  echo "$out step=$st elapsed=$(ps -o etime= -p $pid | tr -d " ")"
done
date
