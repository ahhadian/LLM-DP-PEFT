from transformers import get_linear_schedule_with_warmup
import logging
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import wandb
import math
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager


class Trainer:
    def __init__(self, cfg, model, train_loader, checkpoint=None):
        self.criterion = CrossEntropyLoss()
        self.val_criterion = CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=cfg.optimizer_eps)
        self.gradient_accumulation_steps = cfg.virtual_batch_size // cfg.batch_size
        total_steps = math.ceil(len(train_loader) / self.gradient_accumulation_steps) * cfg.epochs

        if cfg.scheduler:
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=cfg.scheduler_warmup_ratio*total_steps, num_training_steps=total_steps)
        
        self.dp = cfg.dp
        self.model = model
        self.cfg = cfg

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
                epochs=cfg.epochs,
                max_grad_norm=cfg.clipping_threshold,
            )
    
    def train_step(self, train_loader):
        train_loss = 0

        self.model.train()
        self.optimizer.zero_grad()

        if self.cfg.dp:
            with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=self.cfg.batch_size, optimizer=self.optimizer) as new_data_loader:
                for batch_number, batch in tqdm(enumerate(new_data_loader, 1), total=len(new_data_loader)):
                    # Move batch tensors to the same device as the model
                    batch = {k: v.to(self.cfg.device) for k, v in batch.items()}

                    # Forward pass
                    outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                    loss = self.criterion(outputs.logits, batch["label"])

                    loss.backward()
                    train_loss += loss.mean().item()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.cfg.scheduler:
                        self.scheduler.step()
        else:
            for batch_number, batch in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
                # Move batch tensors to the same device as the model
                batch = {k: v.to(self.cfg.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = self.criterion(outputs.logits, batch["label"])

                loss.backward()
                train_loss += loss.mean().item()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.cfg.scheduler:
                    self.scheduler.step()

        return train_loss/len(train_loader)

    def evaluate_step(self, val_loader):
        # Evaluation loop
        val_loss = 0
        self.model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader):
                # Move batch tensors to the same device as the model
                batch = {k: v.to(self.cfg.device) for k, v in batch.items()}

                # Forward pass and compute validation loss
                outputs = self.model(batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = self.val_criterion(outputs.logits, batch["label"])
                _, preds = torch.max(outputs.logits, dim=1)
                predictions.extend(preds.tolist())
                true_labels.extend(batch["label"].tolist())
        
                val_loss += loss.item()
            accuracy = accuracy_score(true_labels, predictions)
            
            return accuracy , val_loss/len(val_loader)

    def train_and_evaluate(self, epochs, train_loader, val_loader_one, val_loader_two):
        best_accuracy = 0
        best_accuracy_two = 0

        wandb_log = []

        for epoch in range(epochs):
            log_data = {}
            train_loss = self.train_step(train_loader)
            log_data["train_loss"] = train_loss
            logging.info(f"Epoch {epoch+1} Training loss: {train_loss}")
            accuracy, val_loss = self.evaluate_step(val_loader=val_loader_one)
            log_data["validation_loss"] = val_loss
            log_data["accuracy"] = accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            logging.info(f"Epoch {epoch+1} Validation loss: {val_loss}")
            logging.info(f"Accuracy on validation set: {accuracy * 100} %")
            if val_loader_two:
                accuracy_two , val_loss_two = self.evaluate_step(val_loader=val_loader_two)
                log_data["validation_two_loss"] = val_loss_two
                log_data["accuracy_two"] = accuracy_two
                if accuracy_two > best_accuracy_two:
                    best_accuracy_two = accuracy_two
                logging.info(f"Epoch {epoch+1} Validation two loss: {val_loss_two}")
                logging.info(f"Accuracy on validation two set: {accuracy_two * 100} %")
            
            wandb_log.append(log_data)
            
        logging.info("Best results:")
        
        if self.cfg.dp:
            logging.info(self.privacy_engine.accountant.get_epsilon(delta=self.cfg.delta))
        logging.info(f"Best validatin accuracy: {best_accuracy}")
        if val_loader_two:
            logging.info(f"Second validation set accuracy: {best_accuracy_two}")
        if self.cfg.use_wandb:
            for i, epoch_data in enumerate(wandb_log):
                wandb.log(epoch_data)
