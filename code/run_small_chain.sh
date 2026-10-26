#!/bin/bash
# E10 other learners. MovieLens MF in a noisy regime (batch 256; at batch 4096 the lr "temperature" barely moves the loss),
# CIFAR-10 CNN, each with momentum SGD and with plain SGD.
C=/home/soso/advsci/code/kov; L=/home/soso/advsci/logs/kov
bash $C/run_small2.sh mf mfn 0.5 1.5 0.0625 0.75 20000 "--opt sgd --mom 0.9 --wd 1e-4 --bs 256" > $L/run_mfn.log 2>&1
bash $C/run_small2.sh cifar cifar 0.04 0.12 0.005 0.06 4000 "--opt sgd --mom 0.9 --wd 5e-4 --bs 256" > $L/run_cifar.log 2>&1
bash $C/run_small2.sh mf mf0n 5 15 0.625 7.5 20000 "--opt sgd --mom 0 --wd 1e-4 --bs 256" > $L/run_mf0n.log 2>&1
bash $C/run_small2.sh cifar cifar0 0.4 1.2 0.05 0.6 4000 "--opt sgd --mom 0 --wd 5e-4 --bs 256" > $L/run_cifar0.log 2>&1
echo ALLDONE
