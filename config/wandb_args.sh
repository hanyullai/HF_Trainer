export WANDB_PROJECT='Example'
export WANDB_KEY='YOUR_WANDB_KEY'
export WANDB_RUN_NAME=${JOBID}
# export WANDB_MODE="offline" # not sync with w&b cloud

wandb_options=" \
    --report_to wandb \
    --run_name ${JOBID}
"

wandb_disable_options=" \
    --report_to none
"

# report_to none