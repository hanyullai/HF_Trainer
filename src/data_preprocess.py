from .arguments import model_args, data_args, training_args
from .template import build_prompt
import logging
logger = logging.getLogger(__name__)

def seq2seq_preprocess_eval(example, tokenizer, data_column, data_args):

    prompt_column = data_column['prompt']
    response_column = data_column['response']
    system_column = data_column['system'] if 'system' in data_column else None
    history_column = data_column['history'] if 'history' in data_column else None
    additional = data_column['additional'] if 'additional' in data_column else None

    max_len = data_args.max_source_length

    try:
        if example[prompt_column] and example[response_column]:
            query, answer = example[prompt_column], example[response_column]
            history = example[history_column] if history_column is not None else None
            system = example[system_column] if system_column is not None else None

            prompt = ''

            if additional is not None:
                for (key, column) in additional:
                    if column in example:
                        prompt += f"<{key}> {example[column]} </{key}>\n\n"

            prompt += build_prompt[data_args.template](query, history, system)
    
            input_ids = tokenizer(
                text=prompt, 
                add_special_tokens=True, 
                truncation=False,
                padding='max_length',
                max_length=max_len,
            )

            if len(input_ids['input_ids']) > max_len:
                logger.warning(f"Source length ({len(input_ids['input_ids'])}) exceed! Skip the data")
                return None

            labels = tokenizer.encode(
                text=answer, 
                add_special_tokens=False,
                truncation=False,
                padding='max_length',
                max_length=max_len,
            )

            if len(labels) > max_len:
                logger.warning(f"Target length ({len(labels)}) exceed! Skip the data")
                return None

        else:
            logger.warning(f"Missing prompt or response! Skip the data")
            return None
    
    except Exception as e:
        logger.warning(f"Error occurs when processing data: {e}")
        return None
    
    model_inputs = {
        "input_ids": input_ids['input_ids'],
        "labels": labels,
        "attention_mask": input_ids['attention_mask'] 
    }
    
    return model_inputs

def rm_preprocess(example, tokenizer, data_column, data_args):
    
    prompt_column = data_column['prompt']
    response_column = data_column['response']
    reject_column = data_column['reject']
    system_column = data_column['system'] if 'system' in data_column else None
    history_column = data_column['history'] if 'history' in data_column else None
    additional = data_column['additional'] if 'additional' in data_column else None

    max_src_len = data_args.max_source_length
    max_tgt_len = data_args.max_target_length

    max_seq_len = max_src_len + max_tgt_len

    try:
        if example[prompt_column] and example[response_column] and example[reject_column]:
            query, answer, reject = example[prompt_column], example[response_column], example[prompt_column]
            history = example[history_column] if history_column is not None else None
            system = example[system_column] if system_column is not None else None

            prompt = ''

            if additional is not None:
                for (key, column) in additional:
                    if column in example:
                        prompt += f"<{key}> {example[column]} </{key}>\n\n"
            
            prompt += build_prompt[data_args.template](query, history, system)

            answer_ids = tokenizer(
                text=prompt + ' ' + answer,
                add_special_tokens=True, 
                truncation=False,
                padding="max_length",
                max_length=max_seq_len
            )

            if len(answer_ids['input_ids']) > max_seq_len:
                logger.warning(f"answer length ({len(answer_ids)}) exceed! Skip the data")
                return None
            
            reject_ids = tokenizer(
                text=prompt + ' ' + reject,
                add_special_tokens=True, 
                truncation=False,
                padding="max_length",
                max_length=max_seq_len
            )

            if len(reject_ids['input_ids']) > max_seq_len:
                logger.warning(f"reject length ({len(reject_ids)}) exceed! Skip the data")
                return None
        
        else:
            logger.warning(f"Missing prompt or response! Skip the data")
            return None
            
    except Exception as e:
        logger.warning(f"Error occurs when processing data: {e}")
        return None

    model_inputs = {
        "input_ids_chosen": answer_ids['input_ids'],
        "attention_mask_chosen": answer_ids['attention_mask'],
        "input_ids_rejected": reject_ids['input_ids'],
        "attention_mask_rejected": reject_ids['attention_mask']
    }
    
    return model_inputs

#Only concat prompt, left tokenization for DPODataCollatorWithPadding
def dpo_preprocess_train(example, data_column, data_args):

    prompt_column = data_column['prompt']
    chosen_column = data_column['chosen']
    reject_column = data_column['reject']
    system_column = data_column['system'] if 'system' in data_column else None
    history_column = data_column['history'] if 'history' in data_column else None
    additional = data_column['additional'] if 'additional' in data_column else None

    prompt = ''
    if example[prompt_column] and example[chosen_column] and example[reject_column]:
        query, answer, reject = example[prompt_column], example[chosen_column], example[reject_column]
        history = example[history_column] if history_column is not None else None
        system = example[system_column] if system_column is not None else None

        if additional is not None:
            for (key, column) in additional:
                if column in example:
                    prompt += f"<{key}> {example[column]} </{key}>\n\n"
        
        prompt += build_prompt[data_args.template](query, history, system)
    
    else:
        logger.warning(f"Missing prompt, chosen or reject! Skip the data")
        return None

    prompt_input = {
        "prompt": prompt,
        "chosen": answer,
        "rejected": reject
    }
        
    return prompt_input

def seq2seq_preprocess_train(example, tokenizer, data_column, data_args):
    
    prompt_column = data_column['prompt']
    response_column = data_column['response']
    system_column = data_column['system'] if 'system' in data_column else None
    history_column = data_column['history'] if 'history' in data_column else None
    additional = data_column['additional'] if 'additional' in data_column else None

    max_src_len = data_args.max_source_length
    max_tgt_len = data_args.max_target_length
    max_seq_len = max_src_len + max_tgt_len

    try:
        if example[prompt_column] and example[response_column]:
            query, answer = example[prompt_column], example[response_column]
            history = example[history_column] if history_column is not None else None
            system = example[system_column] if system_column is not None else None

            prompt = ''

            if additional is not None:
                for (key, column) in additional:
                    if column in example:
                        prompt += f"<{key}> {example[column]} </{key}>\n\n"
            
            prompt += build_prompt[data_args.template](query, history, system)
            
            a_ids = tokenizer.encode(
                text=prompt,
                add_special_tokens=True, 
                truncation=False,
                padding=False,
                max_length=max_src_len
            )

            if len(a_ids) > max_src_len:
                logger.warning(f"Source length ({len(a_ids)}) exceed! Skip the data")
                return None

            b_ids = tokenizer.encode(
                text=answer, 
                add_special_tokens=False, 
                truncation=False,
                padding=False,
                max_length=max_tgt_len - 1
            ) + [tokenizer.eos_token_id]

            if len(b_ids) > max_tgt_len:
                logger.warning(f"Target length ({len(b_ids)}) exceed! Skip the data")
                return None

            input_ids = a_ids + b_ids

            context_length = len(a_ids)
            if data_args.ignore_source_token_for_loss:
                labels = [-100] * context_length + b_ids
            else:
                labels = a_ids + b_ids
            
            # padding to max length
            if data_args.pad_to_max_length:
                pad_len = max_seq_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len

                label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
                labels = labels + [label_pad_token_id] * pad_len
        
        else:
            logger.warning(f"Missing prompt or response! Skip the data")
            return None
            
    except Exception as e:
        logger.warning(f"Error occurs when processing data: {e}")
        return None

    model_inputs = {
        "input_ids": input_ids,
        "labels": labels,
    }
    
    return model_inputs

def auto_regressive_preprocess_train(example, tokenizer, data_column, data_args):
    
    prompt_column = data_column['prompt']

    max_src_len = data_args.max_source_length    

    try:
        if example[prompt_column]:
            prompt = example[prompt_column]

            input_ids = tokenizer.encode(
                text=prompt, 
                add_special_tokens=True, 
                truncation=False,
                padding=False,
                max_length=max_src_len - 1
            ) + [tokenizer.eos_token_id]

            if len(input_ids) > max_src_len:
                logger.warning(f"Source length ({len(input_ids)}) exceed! Skip the data")
                return None

            labels = input_ids
            
            # padding
            if data_args.pad_to_max_length:
                pad_len = max_src_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len

                label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
                labels = labels + [label_pad_token_id] * pad_len
        
        else:
            logger.warning(f"Missing prompt! Skip the data")
            return None
            
    except Exception as e:
        logger.warning(f"Error occurs when processing data: {e}")
        return None

    model_inputs = {
        "input_ids": input_ids,
        "labels": labels,
    }
        
    return model_inputs

def print_dataset_example(example):
    print(example)