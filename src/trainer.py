from .arguments import model_args, data_args, training_args
from .peft import get_peft_config
from .metrics import get_metric

from transformers import Seq2SeqTrainer
from transformers.utils import is_sagemaker_mp_enabled
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from trl import SFTTrainer, RewardTrainer, DPOTrainer
from transformers import default_data_collator, DataCollatorForSeq2Seq
import logging
import json
from peft import get_peft_model
from copy import deepcopy


logger = logging.getLogger(__name__)

# For setting different lr for different modules
class CustomTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.lr_modules = kwargs.pop('lr_modules', None)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """
        Setup the optimizer.
        """
        if not self.lr_modules:
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = []
            for n, p in opt_model.named_parameters():
                if p.requires_grad:
                    for module_name, lr in self.lr_modules.items():
                        if module_name in n:
                            learning_rate = lr
                            break
                    else:
                        learning_rate = None
                    
                    x = {
                        "params": [p],
                    }

                    if n in decay_parameters:
                        x["weight_decay"] = self.args.weight_decay
                    else:
                        x["weight_decay"] = 0.0

                    if learning_rate is not None:
                        x["lr"] = learning_rate
                    
                    optimizer_grouped_parameters.append(x)

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


def init_trainer(model, tokenizer, train_dataset, eval_dataset):

    # Experiment settings
    do_train = training_args.do_train
    do_eval = training_args.do_eval and not training_args.do_predict
    do_predict = training_args.do_predict
    do_generate = do_predict and training_args.training_stage != 'rm'

    if training_args.lora:
        lora_config = get_peft_config()

    # Data 
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
            max_length=data_args.max_source_length + data_args.max_target_length + 1
        )

    if training_args.training_stage == 'pretrain' or do_generate:
        if training_args.lora:
            model = get_peft_model(model, lora_config)
            model.base_model.model.enable_input_require_grads() # solution for https://github.com/huggingface/peft/issues/522

        # custom optimzizer
        if training_args.lr_module_path:
            lr_modules = json.load(open(training_args.lr_module_path))

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if do_train else None,
            eval_dataset=eval_dataset if do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=get_metric(training_args.metrics, tokenizer) if training_args.predict_with_generate else None
        )

    elif training_args.training_stage == 'sft':
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset if do_train else None,
            eval_dataset=eval_dataset if do_eval else None,
            tokenizer=tokenizer,
            compute_metrics=get_metric(training_args.metrics, tokenizer) if training_args.predict_with_generate else None,
            max_seq_length=data_args.max_source_length + data_args.max_target_length,
            peft_config=lora_config if training_args.lora else None,
            dataset_text_field='text', # not used, just to avoid error
            neftune_noise_alpha=training_args.neftune_alpha
        )
    
    elif training_args.training_stage == 'rm':
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset if do_train else None,
            eval_dataset=eval_dataset if do_eval else None,
            tokenizer=tokenizer,
            compute_metrics=None, #default for accuracy computation
            peft_config=lora_config if training_args.lora else None
        )

    elif training_args.training_stage == 'dpo':
        # ref_model = deepcopy(model)
        trainer = DPOTrainer(
            model=model,
            # ref_model=ref_model,
            beta=training_args.beta,
            args=training_args,
            data_collator=None, # default for DPODataCollatorWithPadding
            label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
            padding_value=tokenizer.pad_token_id,
            train_dataset=train_dataset if do_train else None,
            eval_dataset=eval_dataset if do_eval else None,
            tokenizer=tokenizer,
            max_length=data_args.max_source_length + data_args.max_target_length,
            max_prompt_length=data_args.max_source_length,
            max_target_length=data_args.max_target_length,
            peft_config=lora_config if training_args.lora else None,
            compute_metrics=get_metric(training_args.metrics, tokenizer) if training_args.predict_with_generate else None,
        )
    else:
        raise ValueError('Invalid training stage!')

    logger.info('*** Trainer Initialized! ***')
    return trainer