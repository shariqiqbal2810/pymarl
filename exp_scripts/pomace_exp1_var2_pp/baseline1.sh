#!/bin/bash
# To be called from inside Docker container for now

#cd /pymarl/src
export PYTHONPATH=$PYTHONPATH:/pymarl/src

TAG=$1
TIMESTAMP=$2
N_REPEAT=$3
N_GPUS=`nvidia-smi -L | wc -l`
N_UPPER=`expr $N_GPUS - 1`

for i in $(seq 1 $N_REPEAT); do
  GPU_ID=`shuf -i0-${N_UPPER} -n1`
  echo "Starting repeat number $i on GPU $GPU_ID"
  NV_GPU=${GPU_ID} ../.././docker.sh python3 /pymarl/src/main.py ---exp_name="pomace_exp1_coma_pp"  with name=${TAG}__pomace_exp1_coma_pp_${TIMESTAMP}__repeat${i} pomace_exp_variant=2 use_tensorboard=False &
  sleep 10s
done
