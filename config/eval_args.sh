#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)
EVAL_INTERVAL=50

eval_loss_options=" \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps ${EVAL_INTERVAL} \
    --per_device_eval_batch_size 4 \
"

eval_generate_options=" \
    --do_predict \
    --predict_with_generate \
    --metrics rouge \
    --evaluation_strategy steps \
    --eval_steps ${EVAL_INTERVAL} \
    --per_device_eval_batch_size 4 \
    --max_predict_samples 100 \
"

#   --generation_max_length 128 \
#   --generation_num_beams 5 \``