#!/bin/bash
# To be called from inside Docker container for now

#cd /pymarl/src
export PYTHONPATH=$PYTHONPATH:/pymarl/src

filename=$(basename -- "$0")
extension="${filename##*.}"
filename="${filename%.*}"

TAG=$1
N_REPEAT=$2
TIMESTAMP=`date "+%d/%m/%y-%H:%M:%S"`
N_GPUS=`nvidia-smi -L | wc -l`
N_UPPER=`expr $N_GPUS - 1`

for i in $(seq 1 $N_REPEAT); do
  GPU_ID=`shuf -i0-${N_UPPER} -n1`
  echo "Starting repeat number $i on GPU $GPU_ID"
  NV_GPU=${GPU_ID} ../../.././docker.sh python3 /pymarl/src/main.py --exp_name="flounderl/moneyshot_5m" with name=${TAG}__${filename}__${TIMESTAMP}__repeat${i} use_tensorboard=False env_args.fully_observable=False flounderl_fix_level1_pair=True flounderl_epsilon_start_level1=0.0 flounderl_delegate_if_zero_ck=True use_hdf_logger=True save_episode_samples=True save_model=True save_model_interval=100000 &
  sleep 10s
done