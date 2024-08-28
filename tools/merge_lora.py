import torch
import argparse
from peft import PeftModel
from transformers import AutoModel


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_model_dir', type=str, help='the original model directory')
    parser.add_argument('--model_dir', type=str, help='the peft model directory')
    parser.add_argument('--output_dir', type=str, help='the output model directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = set_args()
    base_model = AutoModel.from_pretrained(args.ori_model_dir, torch_dtype=torch.float16, trust_remote_code=True)
    lora_model = PeftModel.from_pretrained(base_model, args.model_dir, torch_dtype=torch.float16)
    lora_model.to("cpu")
    model = lora_model.merge_and_unload()
    model.save_pretrained(args.output_dir)