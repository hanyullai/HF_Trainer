#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

model_options=" \
    --model_name_or_path YOUR_MODEL_PATH \
    --bf16 \
"