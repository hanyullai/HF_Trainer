from .arguments import model_args, data_args, training_args

def push_to_hub(trainer):
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": training_args.training}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    trainer.push_to_hub(**kwargs)