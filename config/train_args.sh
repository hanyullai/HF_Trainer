#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

DEFAULT_DEEPSPEED_CONFIG="${script_dir}/ds_config/ds_config_zero2.json"

train_options=" \
    --do_train \
    --training_stage sft \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --warmup_ratio 0.01 \
    --gradient_checkpointing true \
    --gradient_accumulation_steps 1 \
    --deepspeed ${DEFAULT_DEEPSPEED_CONFIG} \

    --logging_strategy steps \
    --logging_steps 1 \
    
    --seed 42 \
    --lr_scheduler_type cosine \
"

sft_options=" \
    --neftune_alpha 0 \
"

dpo_options=" \
    --beta 0.1 \
"

train_options=" \
    ${train_options} ${sft_options} ${dpo_options}
"