from src.arguments import model_args, data_args, training_args
from src.logging import init_report, init_logging
from src.data_load import load_data
from src.trainer import init_trainer
from src.model_init import initialize_model_and_tokenizer
from src.hub import push_to_hub
from src.run_exp import run_exp
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # disable warning
    
def main():
    # Init logging
    init_logging()

    # Init training record like tensorboard, wandb
    init_report()

    # Init model and tokenizer
    model, tokenizer, last_checkpoint = initialize_model_and_tokenizer()

    # Load dataset
    train_dataset, eval_dataset, predict_dataset = load_data(tokenizer)

    # Init trainer
    trainer = init_trainer(model, tokenizer, train_dataset, eval_dataset)

    # Run experiment
    run_exp(trainer, tokenizer, train_dataset, eval_dataset, predict_dataset, last_checkpoint)
    
    if training_args.push_to_hub:
        push_to_hub(trainer)


if __name__ == "__main__":
    main()
