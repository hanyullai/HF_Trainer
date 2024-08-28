from .arguments import model_args, data_args, training_args
from peft import LoraConfig

def get_peft_config():
    lora_module_name = training_args.lora_module_name.split(",")
    modules_to_save = []
    lora_config = LoraConfig(
        r=training_args.lora_dim,
        lora_alpha=training_args.lora_alpha,
        target_modules=lora_module_name,
        lora_dropout=training_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
        inference_mode=False
    )
    
    return lora_config