#!/bin/sh

export JOBID=`date '+%Y_%m_%d_%H_%M_%S'`

# config
source config/machine_config.sh
source config/model_args.sh
source config/data_args.sh
source config/eval_args.sh
source config/save_args.sh
source config/wandb_args.sh
source config/gen_args.sh

# run command
DISTRIBUTED_ARGS="torchrun --nnodes ${NUM_WORKERS} --node_rank ${NODE_RANK} --nproc_per_node ${NUM_GPUS_PER_WORKER} \
        --master_addr localhost --master_port ${MASTER_PORT}"

run_cmd="${DISTRIBUTED_ARGS} run.py"
run_cmd+=$model_options
run_cmd+=$data_options
run_cmd+=$eval_generate_options
run_cmd+=$save_options
run_cmd+=$wandb_disable_options
run_cmd+=$gen_options

echo $run_cmd 2>&1 | tee ./logs/${JOBID}.log
eval $run_cmd 2>&1 | tee -a ./logs/${JOBID}.log