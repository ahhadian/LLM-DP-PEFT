import argparse
import torch
from media import *

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_arguments()
        self.args = self.parse()
        self.post_process()
    
    def parse(self):
        return self.parser.parse_args()
    
    def add_arguments(self):
        self.parser.add_argument('--device', type=int, default=0, help='Device number to use for training')
        # self.parser.add_argument('--gpu_count', type=int, default=1, help='Number of GPUs available')
        self.parser.add_argument('--seed', type=int, default=1234, help='Set seed for reproducability')

        self.parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
        self.parser.add_argument('--virtual_batch_size', type=int, default=8, help='batch size for updating model parameters')
        self.parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
        self.parser.add_argument('--lr', type=float, default='2e-3', help='Learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for optimizer')
        self.parser.add_argument('--optimizer_eps', type=float, default=1e-6, help='optimizer eps')
        self.parser.add_argument("--scheduler", type=int, default=1, help="Uses scheduler if 1")
        self.parser.add_argument("--scheduler_type", type=str, default="linear", choices=['linear', 'steplr'], help="Scheduler types")
        self.parser.add_argument('--scheduler_warmup_ratio', type=float, default=0.06, help='Scheduler warmup ratio * total steps = warmup steps')
        self.parser.add_argument('--scheduler_warmup_steps', type=int, default=None, help='Warmup steps can be given directly')
        self.parser.add_argument('--scheduler_step_size', type=int, default=1, help='Scheduler step size for stepLR scheduler')
        self.parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Scheduler decrease rate for stepLR scheduler')

        self.parser.add_argument('--model_name', type=str, default='gpt2', choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], help='PEFT mode for fine-tuning')
        self.parser.add_argument('--seq_length', type=int, default=128, help='Max length for tokenization')
        self.parser.add_argument('--peft_mode', type=str, default='bitfit', choices=['lora', 'bitfit', 'full', 'lorabitfit', 'adapter', 'adapterbitfit'], help='PEFT mode for fine-tuning')
        self.parser.add_argument('--rank', type=int, default=8, help='Rank for lora')
        self.parser.add_argument('--alpha', type=int, default=16, help='Alpha for lora')
        self.parser.add_argument('--drop_out', type=float, default=0.0, help='Dropout for lora')
        self.parser.add_argument('--reduction_factor', type=int, default=16, help='Reduction_factor for adapter')
        self.parser.add_argument('--dataset', type=str, default='e2e_nlg', choices=['e2e_nlg', 'dart'], help='Dataset name')
        self.parser.add_argument('--toy_example', type=int, default=0, help='if 1, the first 1024 data from train dataset will be used for training')

        self.parser.add_argument("--dp", type=int, default=0, help="Fine-tune using differential privacy if 1")
        self.parser.add_argument("--epsilon", type=int, default=3, help="Epsilon in privacy budget")
        self.parser.add_argument("--delta", type=float, default=1e-5, help="Delta in privacy budget")
        self.parser.add_argument('--clipping_mode', type=str, default='default', choices=['default', 'ghost'], help='Clipping mode for DP fine-tuning')
        self.parser.add_argument("--clipping_threshold", type=float, default=0.1, help="Max grad norm")

        self.parser.add_argument("--use_wandb", type=int, default=0, help="Uses wandb if 1")
        self.parser.add_argument("--wandb_project_name", type=str, default="Project-DP", help="Wandb project name")
        self.parser.add_argument("--run_name", type=str, default=None, help="run name")

        self.parser.add_argument("--beam_size", type=int, default=5, help="Number of beans for generation")

        self.parser.add_argument('--f', type=str, default=None, help='Path to Jupyter kernel JSON file')

        self.parser.add_argument("--two_step_training", type=int, default=0, help="if 1, first finetunes adapter or lora then bitfit")
        self.parser.add_argument('--lr_two', type=float, default='2e-3', help='Learning rate for second step of training')
        self.parser.add_argument('--virtual_batch_size_two', type=int, default=8, help='batch size for updating model parameters for scond step of training')
        self.parser.add_argument('--epochs_two', type=int, default=5, help='Number of epochs for second step training')
        self.parser.add_argument('--weight_decay_two', type=float, default=0.1, help='Weight decay for second optimizer')

    
    def post_process(self):
        assert self.args.virtual_batch_size % self.args.batch_size == 0, "virtual_batch_size should be devisible by batch_size"
        self.args.device = torch.device(f'cuda:{self.args.device}' if torch.cuda.is_available() else "cpu")
        self.args.media_path = media_path
        self.args.model_cache_path = model_cache_path