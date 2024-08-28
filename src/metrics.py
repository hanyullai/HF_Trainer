from .arguments import model_args, data_args, training_args
import rouge
import rouge_chinese
import jieba
import numpy as np
import re

# Rouge Metric
def compute_rouge(eval_preds, tokenizer):
    def _check_contain_chinese(check_str):
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fa5':
                return True
        return False

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        pred = pred.split('答：')[-1].strip() # get answer part
        if _check_contain_chinese(label):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            _rouge = rouge_chinese.Rouge()
            scores = _rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
        else:
            _rouge = rouge.Rouge()
            scores = _rouge.get_scores(pred, label)

        result = scores[0]
        
        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))

    return score_dict

def compute_exact_match(eval_preds, tokenizer):

    def exact_match_score(pred, label):
        return pred.lower() == label.lower()

    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {
        "exact_match": []
    }

    for pred, label in zip(decoded_preds, decoded_labels):
        pred = pred.split('答：')[-1] # get answer part
        score_dict["exact_match"].append(int(exact_match_score(pred, label)))

    score_dict["exact_match"] = float(np.mean(score_dict["exact_match"]))
    
    return score_dict

def compute_include_label(eval_preds, tokenizer):

    def include_label_score(pred, label):
        return label.lower() in pred.lower()

    preds, labels = eval_preds
    
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {
        "include_label": []
    }

    for pred, label in zip(decoded_preds, decoded_labels):
        pred = pred.split('答：')[-1] # get answer part
        score_dict["include_label"].append(int(include_label_score(pred, label)))

    score_dict["include_label"] = float(np.mean(score_dict["include_label"]))
    
    return score_dict

# Compute Operation
def compute_operation(eval_preds, tokenizer):

    def _extract_operation(answer):
        operation = [
            r"(#?Click#?)\s*([A-Z]{2})",
            r"(#?Type#?)\s*([A-Z]{2})\s*([\w\s]+)",
            r"(#?Scroll up#?)\s*(\d+\.?\d*)",
            r"(#?Scroll down#?)\s*(\d+\.?\d*)",
            r"(#?Goto#?)\s*(https?:\/\/[-a-z0-9]+(?:\.[-a-z0-9]+)*\.(?:com|cn|edu|uk)(?:\/[-a-z0-9_:@&?=+,.!/~*'%$]*)?)",
            r"(#?Go backward#?)",
            r"(#?Go forward#?)",
            r"(#?Hover#?)\s*([A-Z]{2})",
            r"(#?Answer#?)\s*([\w\s]+)",
            r"(#?Login#?)",
            r"(#?Verify#?)",
            r"(#?Exit#?)"
        ]
        for regex in operation:
            matches = re.findall(regex, answer)

            if matches:
                m = matches[-1]
                if isinstance(m, tuple):
                    operation = m[0]
                    param = list(m[1:])
                    for i in range(len(param)):
                        param[i] = param[i].lower().strip('\'"“”')
                else:
                    operation = m
                    param = None
                
                operation = operation.lower().strip('#')
                    
                return operation, param
            
        return None, None

    def _check_contain_chinese(check_str):
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fa5':
                return True
        return False

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    score_dict = {
        "op_acc": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        pred = pred.split('答：')[-1] # get answer part
        
        pred_op, pred_param = _extract_operation(pred)
        label_op, label_param = _extract_operation(label)
        
        if not pred_op:
            success = 0
        else:
            success = 100 if pred_op == label_op and pred_param == label_param else 0

        score_dict['op_acc'].append(success)

    score_dict['op_acc'] = float(np.mean(score_dict['op_acc']))

    return score_dict


metrics = {
    'rouge': compute_rouge,
    'operation': compute_operation,
    'exact_match': compute_exact_match,
    'include_label': compute_include_label,
}

def get_metric(metric_names, tokenizer):
    global metrics
    metric_names = metric_names.split(',')

    def compute_metrics(eval_preds):
        score_dict = {}
        for metric in metric_names:
            if metric in metrics:
                score = metrics[metric](eval_preds, tokenizer)
                score_dict.update(score)
        return score_dict

    return compute_metrics