from .arguments import model_args, data_args, training_args
from .data_preprocess import (
    seq2seq_preprocess_eval,
    seq2seq_preprocess_train,
    rm_preprocess,
    dpo_preprocess_train,
    auto_regressive_preprocess_train,
    print_dataset_example
)

from datasets import load_dataset
from functools import partial
import json
import logging
import torch
from time import time

logger = logging.getLogger(__name__)

class SeqioDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, map_fun, max_num_examples=None):
        super(SeqioDataset, self).__init__()
        self.dataset = dataset
        self.map_fun = map_fun
        try:
            self.num_examples = len(dataset)
        except TypeError:
            self.num_examples = 10 ** 9 if max_num_examples is None else max_num_examples

    def __iter__(self):
        for ix, data in enumerate(self.dataset):
            processed_data = self.map_fun(data)
            if processed_data:
                yield processed_data
            else:
                logger.warning(f"Data {ix} Preprocess Error")

    def __getitem__(self, key):
        processed_data = self.map_fun(self.dataset[key])
        if processed_data:
            return processed_data
        else:
            logger.warning(f"Data {key} Preprocess Error")
            return None

    def __len__(self):
        return self.num_examples

def load_data(tokenizer):
    # Experiment settings
    do_train = training_args.do_train
    do_eval = training_args.do_eval and not training_args.do_predict
    do_predict = training_args.do_predict

    # load datasets
    data_files = {}
    if do_train and data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if do_eval and data_args.validation_file is not None:
        data_files["dev"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if do_predict and data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.

    # Get the column names for input/target.
    data_column = json.load(open(data_args.data_column_path))
    if training_args.training_stage == 'pretrain':
        preprocess_function_train = partial(
            auto_regressive_preprocess_train,
            tokenizer=tokenizer,
            data_column=data_column,
            data_args=data_args
        )
    elif training_args.training_stage == 'sft':
        preprocess_function_train = partial(
            seq2seq_preprocess_train,
            tokenizer=tokenizer,
            data_column=data_column,
            data_args=data_args
        )
    elif training_args.training_stage == 'rm':
        preprocess_function_train = partial(
            rm_preprocess,
            tokenizer=tokenizer,
            data_column=data_column,
            data_args=data_args
        )
    elif training_args.training_stage == 'dpo':
        preprocess_function_train = partial(
            dpo_preprocess_train,
            data_column=data_column,
            data_args=data_args
        )   
    else:
        logger.warning('Error: training not found')
        raise
    
    preprocess_function_eval = partial(
        seq2seq_preprocess_eval,
        tokenizer=tokenizer,
        data_column=data_column,
        data_args=data_args
    )

    train_dataset, eval_dataset, predict_dataset = None, None, None
    if do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        # Preload
        if data_args.preload_data:
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                t0 = time()
                train_dataset = train_dataset.map(
                    preprocess_function_train,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
                logger.info("train data preprocessing time cost:", time() - t0)
        else:
            # Iterable
            # TODO DPO not support iterable
            train_dataset = SeqioDataset(train_dataset, preprocess_function_train, data_args.max_train_samples)
        
        # print_dataset_example(train_dataset[0])

    if do_eval:
        if "dev" not in raw_datasets:
            raise ValueError("--do_eval requires a eval dataset")
        eval_dataset = raw_datasets["dev"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        if training_args.predict_with_generate:
            preprocess_fun = preprocess_function_eval
        else:
            preprocess_fun = preprocess_function_train
            
        # Preload
        if data_args.preload_data:
            with training_args.main_process_first(desc="eval dataset map pre-processing"):
                t0 = time()
                eval_dataset = eval_dataset.map(
                    preprocess_fun,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on eval dataset",
                )
                logger.info("eval data preprocessing time cost:", time() - t0)
        else:
            # Iterable
            eval_dataset = SeqioDataset(eval_dataset, preprocess_fun, data_args.max_eval_samples)

    if do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

        # Preload
        if data_args.preload_data:
            with training_args.main_process_first(desc="test dataset map pre-processing"):
                t0 = time()
                predict_dataset = predict_dataset.map(
                    preprocess_function_eval,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on predict dataset",
                )
                logger.info("predict data preprocessing time cost:", time() - t0)
        else:
            # Iterable
            predict_dataset = SeqioDataset(predict_dataset, preprocess_function_eval, data_args.max_predict_samples)

    return train_dataset, eval_dataset, predict_dataset