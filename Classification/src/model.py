from transformers import RobertaForSequenceClassification, RobertaTokenizer
import logging
import torch


def prepare_model(cfg):
    tokenizer = RobertaTokenizer.from_pretrained(f"{cfg.media_path}models/roberta-large-tokenizer")
    model = RobertaForSequenceClassification.from_pretrained(f"{cfg.media_path}models/roberta-large-model")
    if cfg.dataset == 'mnli':
        model.classifier.out_proj = torch.nn.Linear(model.classifier.out_proj.in_features, 3, bias=True)
    # adjust model parameters
    if cfg.peft_mode == "lora":
        mutate_model(model.roberta, rank=cfg.rank, alpha=cfg.alpha)
        freeze_non_LoRA(model.roberta, peft_key='sharif_llm')
        logging.info("LoRA model loaded")
    elif cfg.peft_mode == "bitfit":
        for a, b in model.roberta.named_parameters():
            if not 'bias' in a:
                b.requires_grad = False
        logging.info("BiTFiT model loaded")
    elif cfg.peft_mode == "lorabitfit":
        mutate_model(model.roberta, rank=cfg.rank, alpha=cfg.alpha)
        freeze_non_LoRA(model.roberta, peft_key='sharif_llm')
        if cfg.two_step_training == 0:
            for a, b in model.roberta.named_parameters():
                if 'bias' in a:
                    b.requires_grad = True
        logging.info("LoRA and BiTFiT combined model loaded")
    elif cfg.peft_mode == "full":
        logging.info("Full model loaded")
    else:
        logging.info("No acceptable model to load")
    model.to(cfg.device)
    return model, tokenizer


class LoRALayer(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Linear,
        rank: int ,
        alpha: float
        ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = self.alpha / self.rank # scaling factor
        self.in_dim = module.in_features
        self.out_dim = module.out_features
        self.pretrained = module

        self.sharif_llm_A = torch.nn.Linear(self.in_dim, self.rank, bias=False)
        torch.nn.init.kaiming_normal_(self.sharif_llm_A.weight)
        self.sharif_llm_B = torch.nn.Linear(self.rank, self.out_dim, bias=False)
        torch.nn.init.zeros_(self.sharif_llm_B.weight)

    def forward(self, x: torch.Tensor):
        
        pretrained_out = self.pretrained(x)
        lora_out = self.sharif_llm_A(x) # x@A
        lora_out = self.sharif_llm_B(lora_out) # x@A@B
        lora_out = self.scaling * lora_out # Scale by the scaling factor
    
        return pretrained_out + lora_out # x@W + x@A@B*(scaling_factor)
    

def mutate_model(model: torch.nn.Module, rank: int, alpha: float):
    """
    Replaces all linear layers in the model with LoRALinear layers.
    Freeze all params except LoRA params.
    """
    # make sure there are no LoRALayer is in the model; return if there are any
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            logging.info("Model already contains LoRALinear layers! \n Try reloading the model.")
            return

    # we want to replace all query and value Linear modules with LoRALayer
    for name, module in model.named_children():
        # if the module is linear and the name is for query or value
        if isinstance(module, torch.nn.Linear) and (name == 'query' or name == 'value'):
            # replace the module with LoRALayer
            lora_layer = LoRALayer(module, rank, alpha)
            setattr(model, name, lora_layer)
        else:
            mutate_model(module, rank, alpha) # recursively call the function on the module


def freeze_non_LoRA(model, peft_key):
    for param_name, weights in model.named_parameters():
        weights.requires_grad = peft_key in param_name
