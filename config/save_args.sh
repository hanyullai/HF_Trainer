#! /bin/bash
export STORE_DIR=YOUR_SAVE_PATH/${JOBID}
mkdir -p ${STORE_DIR}

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)
SAVE_INTERVAL=$((EVAL_INTERVAL * 1))

save_options=" \
    --save_strategy steps \
    --save_steps ${SAVE_INTERVAL} \
    --output_dir ${STORE_DIR} \
"

# For continue training
# --overwrite_output_dir \

# --load_best_model_at_end \
# --metric_for_best_model loss \
# --greater_is_better false
# --save_total_limit 10 \