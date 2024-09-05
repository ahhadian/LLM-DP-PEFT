from transformers import get_linear_schedule_with_warmup
import logging
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager


import wandb
import math
from model import save_model
from torch.optim.lr_scheduler import StepLR


class Trainer:
    def __init__(self, cfg, model, train_loader, checkpoint=None, second_trainer=False):
        if second_trainer:
            self.epochs = cfg.epochs_two
            self.lr = cfg.lr_two
            self.weight_decay = cfg.weight_decay_two
        else:
            self.epochs = cfg.epochs
            self.lr = cfg.lr
            self.weight_decay = cfg.weight_decay
        self.optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=cfg.optimizer_eps)
        self.gradient_accumulation_steps = cfg.virtual_batch_size // cfg.batch_size
        total_steps = len(train_loader) * self.gradient_accumulation_steps * self.epochs
        
        if cfg.scheduler:
            if cfg.scheduler_type == "linear":
                warmup_steps = cfg.scheduler_warmup_steps if cfg.scheduler_warmup_steps else cfg.scheduler_warmup_ratio*total_steps
                self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            elif cfg.scheduler_type == "steplr":
                self.scheduler = StepLR(self.optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
        
        self.dp = cfg.dp
        self.model = model
        self.cfg = cfg
        self.save_path = f"{cfg.media_path}generation_saved_models/{cfg.dataset}/{cfg.peft_mode}"
        self.model_name = self.cfg.run_name if self.cfg.run_name else "best_model"

        if cfg.dp:
            self.model.train()
            self.privacy_engine = PrivacyEngine(
                accountant="rdp",
            )
            if checkpoint:
                self.privacy_engine.load_checkpoint(path=checkpoint, module=self.model)
            self.model, self.optimizer, _ = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=train_loader,
                target_epsilon=cfg.epsilon,
                target_delta=cfg.delta,
                epochs=self.epochs,
                max_grad_norm=cfg.clipping_threshold,
            )

    
    def train_step(self, train_loader):
        train_loss = 0

        self.model.train()
        self.optimizer.zero_grad()

        if self.dp:
            with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=self.cfg.batch_size, optimizer=self.optimizer) as new_data_loader:
                for batch_number, batch in tqdm(enumerate(new_data_loader, 1), total=len(new_data_loader)):
                    # Move batch tensors to the same device as the model
                    batch = prepare_inputs(batch)
                    batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    train_loss += loss.item()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.cfg.scheduler and self.cfg.scheduler_type == "linear":
                        self.scheduler.step()

                if self.cfg.scheduler and self.cfg.scheduler_type == "steplr":
                    self.scheduler.step()
        else:
            for batch_number, batch in tqdm(enumerate(new_data_loader, 1), total=len(new_data_loader)):
                # Move batch tensors to the same device as the model
                batch = prepare_inputs(batch)
                batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.cfg.scheduler and self.cfg.scheduler_type == "linear":
                    self.scheduler.step()

            if self.cfg.scheduler and self.cfg.scheduler_type == "steplr":
                self.scheduler.step()

        return train_loss/len(train_loader)

    def evaluate_step(self, val_loader):
        # Evaluation loop
        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader):
                # Move batch tensors to the same device as the model
                batch = prepare_inputs(batch)
                batch = {k: v.to(self.cfg.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = compute_loss_per_input(outputs, batch)
        
                val_loss += loss.mean().item()
            
        return val_loss/len(val_loader)

    def train_and_evaluate(self, epochs, train_loader, val_loader):
        best_validation_loss = None
        best_epoch = 0

        wandb_log = []

        for epoch in range(epochs):
            log_data = {}
            train_loss = self.train_step(train_loader)
            log_data["train_loss"] = train_loss
            logging.info(f"Epoch {epoch+1} Training loss: {train_loss}")
            val_loss = self.evaluate_step(val_loader=val_loader)
            log_data["validation_loss"] = val_loss
            logging.info(f"Epoch {epoch+1} Validation loss: {val_loss}")
            if best_validation_loss is None or val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_epoch = epoch
                save_model(self.model, self.cfg.peft_mode, self.save_path, self.model_name)
                logging.info(f"Model improved and saved for epoch {epoch+1}")
            
            wandb_log.append(log_data)
            
        logging.info("Best results:")
        if self.cfg.dp:
            logging.info(self.privacy_engine.accountant.get_epsilon(delta=self.cfg.delta))
        logging.info(f"Best validatin loss: {best_validation_loss} for Epoch: {best_epoch+1}")

        if self.cfg.use_wandb:
            for i, epoch_data in enumerate(wandb_log):
                wandb.log(epoch_data)


def prepare_inputs(batch):
    batch.pop('src_attn', None)
    batch.pop('tgt_attn', None)
    batch.pop('src', None)
    return batch

def compute_loss_per_input(outputs, batch):
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    seq_lens = (shift_labels != -100).sum(dim=1)
    loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction="none")
    loss = loss.sum(dim=1) / seq_lens
    return loss

def save_evaluation_output(outputs, path):
    with open(path, "w") as file:
        for strings in outputs:
            for string in strings:
                file.write(string + "\n")
            # file.write("\n")
    file.close()

def generate_evaluation_output(model, tokenizer, data, device, max_length, beam_size=5, do_sample=False, num_return_sequences=1):
    generated_texts = []

    prev = None

    for entry in tqdm(data):
        if prev != entry["meaning_representation"]:
            prev = entry["meaning_representation"]
            prompt = f"{entry['meaning_representation']} {tokenizer.eos_token}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs,
                                        num_beams=beam_size, 
                                        max_length=max_length, 
                                        do_sample=do_sample, 
                                        early_stopping=True, 
                                        min_length=5, 
                                        num_return_sequences=num_return_sequences,
                                        bad_words_ids = [[628], [198], [tokenizer.pad_token_id]],
                                        pad_token_id=tokenizer.eos_token_id,
                                        repetition_penalty=1,
                                        top_k=0,
                                        top_p=0.9)
            
            temp_generated_texts = []
            for output in outputs:
                generated_text = tokenizer.decode(output[len(inputs["input_ids"][0]):], skip_special_tokens=True)
                temp_generated_texts.append(generated_text.strip())
                
            generated_texts.append(temp_generated_texts)
    return generated_texts
        