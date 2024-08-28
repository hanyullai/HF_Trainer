#!/bin/sh

# keep jobid equal between different machine
mkdir -p tmp
if [ "$MLP_ROLE_INDEX" -eq 0 ];
then
    JOBID=`date '+%Y_%m_%d_%H_%M_%S'`
    echo ${JOBID} > tmp/${MLP_WORKER_0_HOST}_${MLP_WORKER_0_PORT}
else
    sleep 20
    JOBID=`cat tmp/${MLP_WORKER_0_HOST}_${MLP_WORKER_0_PORT}`
fi

# config
source config/model_args.sh
source config/data_args.sh
source config/eval_args.sh
source config/save_args.sh
source config/wandb_args.sh
source config/gen_args.sh

# run command
DISTRIBUTED_ARGS="torchrun --nnodes ${MLP_WORKER_NUM} --node_rank ${MLP_ROLE_INDEX} --nproc_per_node ${MLP_GPU} \
        --master_addr ${MLP_WORKER_0_HOST} --master_port ${MLP_WORKER_0_PORT}"

run_cmd="${DISTRIBUTED_ARGS} run.py"
run_cmd+=$model_options
run_cmd+=$data_options
run_cmd+=$eval_generate_options
run_cmd+=$save_options
run_cmd+=$wandb_disable_options
run_cmd+=$gen_options

echo $run_cmd 2>&1 | tee ./logs/${JOBID}.log
eval $run_cmd 2>&1 | tee -a ./logs/${JOBID}.log