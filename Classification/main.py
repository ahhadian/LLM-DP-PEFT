from config import Config
from src.model import prepare_model
from src.data import prepare_data
from src.train import Trainer
import os
import random
import numpy as np
import torch
import wandb
import logging
import transformers
import warnings

warnings.filterwarnings("ignore", "Using a non-full backward hook when the forward contains multiple autograd Nodes ")

transformers.logging.set_verbosity_error()

def set_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    transformers.set_seed(seed)

def copy_model_weights(model1, model2):
    model1.eval()
    model2.eval()
    params1 = model1.parameters()
    params2 = model2.parameters()
    with torch.no_grad():
        for param1, param2 in zip(params1, params2):
            param2.data.copy_(param1.data)

# Returns number of trainbale parameters of the model
def get_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Returns number of parameters of the model
def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main(cfg):
    set_seeds(cfg.seed)

    model, tokenizer = prepare_model(cfg)
    num_of_all_params = get_number_of_parameters(model)
    num_of_trainbale_params = get_number_of_trainable_parameters(model)
    percentage = round(100 * num_of_trainbale_params / num_of_all_params, 2)
    logging.info(f"New Model loaded successfully and number of trainable params is: {num_of_trainbale_params} out of {num_of_all_params}")
    logging.info(f"Percentage of trainable parameters: {percentage} %")
    
    train_loader, val_loader_one, val_loader_two = prepare_data(cfg, tokenizer)
    logging.info("Data is ready")

    trainer = Trainer(cfg, model, train_loader)
    trainer.train_and_evaluate(cfg.epochs, train_loader, val_loader_one, val_loader_two)
    if cfg.two_step_training:
        if cfg.dp:
            trainer.privacy_engine.save_checkpoint(path="temp.pth", module=model)
            model_two, _ = prepare_model(cfg)
            copy_model_weights(model, model_two)
            del model
            model = model_two
        for a, b in model.roberta.named_parameters(): 
            if 'bias' in a:
                b.requires_grad = True
            else:
                b.requires_grad = False
        logging.info("New Model adjusted")
        num_of_all_params = get_number_of_parameters(model)
        num_of_trainbale_params = get_number_of_trainable_parameters(model)
        percentage = round(100 * num_of_trainbale_params / num_of_all_params, 2)
        logging.info(f"New Model loaded successfully and number of trainable params is: {num_of_trainbale_params} out of {num_of_all_params}")
        logging.info(f"Percentage of trainable parameters: {percentage} %")
        
        trainer_two = Trainer(cfg, model, train_loader, checkpoint="temp.pth")
        trainer_two.train_and_evaluate(cfg.epochs, train_loader, val_loader_one, val_loader_two)

    if cfg.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    cfg = Config().args

    log_path = "logs/"
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
    hyperparameters_str = "\n".join([f"{key}: {value}" for key, value in hyperparameters.items()])
    logging.info("config:\n" + hyperparameters_str)

    main(cfg)