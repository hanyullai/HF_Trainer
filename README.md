# HF_Trainer

HF_Trainer is a user-friendly and extensible framework for large-model training based on Huggingface. It supports various mainstream open-source large model training tasks, including pre-training, fine-tuning, reward modeling, and DPO training.

## Usage

### Parameter Configuration

You can find all necessary parameter settings under the `/config` directory, including `data_args.sh` for dataset construction, `train_args.sh` for training, and `eval_args.sh` for evaluation.

To identify parameters that need to be set, search for `YOUR_` in the configuration files. These parameters include `DATASET_PATH`, `MODEL_PATH`, `SAVE_PATH`, and `WANDB_KEY` (if not using W&B, you can disable it in the settings).

If you need to switch between different training stages, you can modify the `training_stage` parameter in `train_args.sh`.

When loading datasets, you can configure the corresponding fields by setting `utils_config/data_column.json`.

### Getting Started

#### Training

We provide two training scripts: `scripts/train_local.sh` for local single-machine training and `scripts/train_submit.sh` for distributed multi-machine training (using PyTorch DDP).

You can start training by running the command: `bash scripts/train_xxxxx.sh`.

Note: To save storage space, the framework automatically deletes optimizer parameters except for the latest checkpoint during training. If you want to save optimizer parameters for all checkpoints, comment out the line `python tools/del_training_state.py ${STORE_DIR} $$ &` in the script.

#### Evaluation

Similar to training, we provide two evaluation scripts for both local and distributed settings. You can start the evaluation by running the command: `bash scripts/eval_xxxxx.sh`.