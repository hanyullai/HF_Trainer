#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

gen_options=" \
    --generation_config ${script_dir}/gen_config/generation_config.json
"