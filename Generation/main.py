from config import Config
import os
import random
import numpy as np
import torch
import wandb
import logging
import transformers
import warnings
import subprocess
from model import load_model, prepare_model, get_number_of_trainable_parameters, load_model_weights, get_number_of_parameters
from data import load_dataset, load_dataloaders
from train import Trainer, generate_evaluation_output, save_evaluation_output
from utils import clean_hyperparameters, copy_model_weights

warnings.filterwarnings("ignore", "Using a non-full backward hook when the forward contains multiple autograd Nodes")

transformers.logging.set_verbosity_error()

def set_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    transformers.set_seed(seed)


def run_metric_script(file_path):
    result = subprocess.run(["e2e/measure_scores.py", "-p", "e2e_ref.txt", file_path], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    lines = output.split('\n')
    return lines[-7:-2]


def main(cfg):
    set_seeds(cfg.seed)

    model, tokenizer = load_model(cfg.model_name, cache_dir=cfg.model_cache_path)
    model = prepare_model(model, cfg)
    num_of_all_params = get_number_of_parameters(model)
    num_of_trainbale_params = get_number_of_trainable_parameters(model)
    percentage = round(100 * num_of_trainbale_params / num_of_all_params, 2)
    logging.info(f"Model loaded successfully and number of trainable params is: {num_of_trainbale_params} out of {num_of_all_params}")
    logging.info(f"Percentage of trainable parameters: {percentage} %")

    dataset = load_dataset(cfg.dataset, cfg.media_path, cfg.toy_example)
    cfg.train_data_size = len(dataset["train"])
    # dataset = tokenize_dataset(tokenizer, dataset, cfg.dataset, cfg.seq_length)
    train_loader, validation_loader = load_dataloaders(dataset, cfg.dataset, cfg.batch_size, cfg.virtual_batch_size, tokenizer, cfg.seq_length, cfg.dp)
    logging.info("Dataset loaded and tokenized")

    trainer = Trainer(cfg, model, train_loader)
    trainer.train_and_evaluate(cfg.epochs, train_loader, validation_loader)

    if cfg.two_step_training and cfg.dp:
        trainer.privacy_engine.save_checkpoint(path="temp.pth", module=model)
        model_two, _ = load_model(cfg.model_name, cache_dir=cfg.model_cache_path)
        model_two = prepare_model(model_two, cfg)
        copy_model_weights(model, model_two)
        del model
        model = model_two
        for a, b in model.named_parameters():
            if 'bias' in a and not 'adapter' in a:
                b.requires_grad = True
            else:
                b.requires_grad = False
        logging.info("New Model adjusted")
        num_of_all_params = get_number_of_parameters(model)
        num_of_trainbale_params = get_number_of_trainable_parameters(model)
        percentage = round(100 * num_of_trainbale_params / num_of_all_params, 2)
        logging.info(f"New Model loaded successfully and number of trainable params is: {num_of_trainbale_params} out of {num_of_all_params}")
        logging.info(f"Percentage of trainable parameters: {percentage} %")
        trainer_two = Trainer(cfg, model, train_loader, second_trainer=True)
        trainer_two.train_and_evaluate(cfg.epochs_two, train_loader, validation_loader)

    # evaluate model on test data
    model.eval()
    model = load_model_weights(model, cfg.peft_mode, f"{trainer.save_path}/{trainer.model_name}.pth")
    evaluation_output = generate_evaluation_output(model, tokenizer, dataset["test"], cfg.device, cfg.seq_length, cfg.beam_size)
    output_path = f"{cfg.media_path}generation_eval_outputs/{cfg.dataset}/{cfg.peft_mode}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_name = cfg.run_name if cfg.run_name else "generation_output"
    save_evaluation_output(evaluation_output, f"{output_path}/{output_name}-v1.txt")
    evaluation_output = generate_evaluation_output(model, tokenizer, dataset["test"], cfg.device, cfg.seq_length, cfg.beam_size, do_sample=True)
    save_evaluation_output(evaluation_output, f"{output_path}/{output_name}-v2.txt")
    logging.info("Generation for test data saved")
    metrics = run_metric_script(f"{output_path}/{output_name}-v1.txt")
    logging.info("Metrics without sampling:")
    for metric in metrics:
        logging.info(metric)
    metrics = run_metric_script(f"{output_path}/{output_name}-v2.txt")
    logging.info("Metrics with sampling:")
    for metric in metrics:
        logging.info(metric)

    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    cfg = Config().args

    log_path = f"logs/{cfg.dataset}/{cfg.peft_mode}/"
    if not os.path.exists(log_path):
         os.makedirs(log_path)
    log_file_name = f"{cfg.run_name}.log" if cfg.run_name else "logs.log"

    if cfg.use_wandb:
        wandb.login(key="YOUR_KEY")
        if cfg.run_name:
            wandb.init(config=cfg, project=f"{cfg.wandb_project_name}-{cfg.dataset}", name=cfg.run_name)
        else:
            wandb.init(config=cfg, project=f"{cfg.wandb_project_name}-{cfg.dataset}")
        log_file_name = wandb.run.name

    logging.basicConfig(filename=f"{log_path}{log_file_name}", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    logging.info("Start of the logging")
    hyperparameters = {key: value for key, value in vars(cfg).items()}
    hyperparameters = clean_hyperparameters(hyperparameters)
    hyperparameters_str = "\n".join([f"{key}: {value}" for key, value in hyperparameters.items()])
    logging.info("config:\n" + hyperparameters_str)

    main(cfg)