#! /bin/bash
export DATASET_PATH=YOUR_DATASET_PATH

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

TRAIN_FILE=${DATASET_PATH}/train.json
VALIDATION_FILE=${DATASET_PATH}/dev.json
TEST_FILE=${DATASET_PATH}/test.json

DATA_COLUMN=${script_dir}/utils_config/data_column.json

data_options=" \
    --dataset_name ${DATASET_PATH}\
    --max_source_length 4096 \
    --max_target_length 1024 \
    --pad_to_max_length \
    --ignore_pad_token_for_loss \
    --ignore_source_token_for_loss \
    --preprocessing_num_workers 16 \
    --train_file ${TRAIN_FILE} \
    --validation_file ${VALIDATION_FILE} \
    --test_file ${TEST_FILE} \
    --data_column_path ${DATA_COLUMN} \
    --template dummy \
    --remove_unused_columns false \
"