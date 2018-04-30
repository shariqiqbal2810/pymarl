#!/bin/bash
# To be called from inside Docker container for now

cd /pymarl/src
export PYTHONPATH=$PYTHONPATH:/pymarl/src

N_REPEAT=$1
N_GPUS=`nvidia-smi -L | wc -l`
N_UPPER=`expr $N_GPUS - 1`

for i in $(seq 1 $N_REPEAT); do
  GPU_ID=`shuf -i0-${N_UPPER} -n1`
  echo "Starting repeat number $i on GPU $GPU_ID"
  NV_GPU=${GPU_ID} ../.././docker.sh python3 /pymarl/src/main.py --exp_name=coma_baseline_2d_3z with name=coma_baseline_2d_3z__repeat${i} &
done
