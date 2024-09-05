from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
import torch
import transformers
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
import os


# Loads model and its tokenizer 
def load_model(model_name, cache_dir="."):
    tokenizer = GPT2Tokenizer.from_pretrained(f"{cache_dir}gpt2/{model_name}-tokenizer")
    model = GPT2LMHeadModel.from_pretrained(f"{cache_dir}gpt2/{model_name}-model")
    add_pad_token(model, tokenizer)
    model.requires_grad_(False)
    return model, tokenizer

# Adds padding token to the tokenizer and model embedding layer
def add_pad_token(model, tokenizer):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    a = model.get_input_embeddings().weight
    a.data[-1] = a.data[:-1].mean(dim=0)

# Returns number of trainbale parameters of the model
def get_number_of_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Returns number of parameters of the model
def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Mutates model structure and adjusts trainable parameters
def prepare_model(model, cfg):
    if cfg.peft_mode == 'bitfit':
        for a, b in model.named_parameters():
            if 'bias' in a:
                b.requires_grad = True
    elif cfg.peft_mode == 'lora':
        model.requires_grad_(True)
        model = convert_gpt2_attention_to_lora(model, cfg.rank, cfg.alpha, cfg.drop_out)
        mark_only_lora_as_trainable(model)
    elif cfg.peft_mode == 'lorabitfit':
        model.requires_grad_(True)
        model = convert_gpt2_attention_to_lora(model, cfg.rank, cfg.alpha, cfg.drop_out)
        mark_only_lora_as_trainable(model)
        if cfg.two_step_training == 0:
            for a, b in model.named_parameters():
                if 'bias' in a:
                    b.requires_grad = True
    elif cfg.peft_mode == 'full':
        model.requires_grad_(True)
    elif cfg.peft_mode == 'adapter':
        model.requires_grad_(False)
        bottleneck_size = model.config.n_embd // cfg.reduction_factor
        mutate_model_adapter(model, bottleneck_size, model.config.n_embd)
        for a, b in model.named_parameters():
            if 'adapter' in a:
                b.requires_grad = True
    elif cfg.peft_mode == 'adapterbitfit':
        model.requires_grad_(False)
        bottleneck_size = model.config.n_embd // cfg.reduction_factor
        mutate_model_adapter(model, bottleneck_size, model.config.n_embd)
        if cfg.two_step_training == 0:
            for a, b in model.named_parameters():
                if 'adapter' in a or 'bias' in a:
                    b.requires_grad = True
        else:
            for a, b in model.named_parameters():
                if 'adapter' in a:
                    b.requires_grad = True
        
    model.to(cfg.device)
    return model

def save_model(model, peft_mode, save_path, model_name):
    if not os.path.exists(save_path):
         os.makedirs(save_path)
    if peft_mode == "bitfit":
        bias_params = {}
        for name, param in model.named_parameters():
            if 'bias' in name:
                 bias_params[name] = param.data.clone()
        torch.save(bias_params, f'{save_path}/{model_name}.pth')
    elif peft_mode == 'lora':
        lora_params = {}
        for name, param in model.named_parameters():
            if 'lora' in name:
                 lora_params[name] = param.data.clone()
        torch.save(lora_params, f'{save_path}/{model_name}.pth')
    elif peft_mode == 'lorabitfit':
        lorabitfit_params = {}
        for name, param in model.named_parameters():
            if 'lora' in name or 'bias' in name:
                 lorabitfit_params[name] = param.data.clone()
        torch.save(lorabitfit_params, f'{save_path}/{model_name}.pth')
    elif peft_mode == 'full':
        pass
    elif peft_mode == 'adapter':
        adapter_params = {}
        for name, param in model.named_parameters():
            if 'adapter' in name:
                 adapter_params[name] = param.data.clone()
        torch.save(adapter_params, f'{save_path}/{model_name}.pth')
    elif peft_mode == 'adapterbitfit':
        adapterbitfit_params = {}
        for name, param in model.named_parameters():
            if 'adapter' in name or 'bias' in name:
                 adapterbitfit_params[name] = param.data.clone()
        torch.save(adapterbitfit_params, f'{save_path}/{model_name}.pth')

def load_model_weights(model, peft_mode, path):
    if peft_mode == 'full':
        pass
    else:
        model_weights = torch.load(path)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in model_weights:
                    param.copy_(model_weights[name])

    return model

# Copyright (c) Xuechen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LoRA layers.

This version does not have merged weights for zero latency inference. It makes the code easier to read and maintain.
Adapted from
    https://github.com/microsoft/LoRA
    https://www.microsoft.com/en-us/research/project/dp-transformers/
"""

class MYDPMergedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        pretrained_module,
        lora_r=0,
        lora_alpha=1.,
        lora_dropout=0.,
    ):
        super(MYDPMergedLinear, self).__init__()
        self.pretrained_module = pretrained_module
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        if self.lora_r > 0:
            self.lora_A = nn.Linear(in_features=in_features, out_features=lora_r, bias=False)
            self.lora_B = nn.Linear(in_features=lora_r, out_features=out_features, bias=False)
            self.scaling = self.lora_alpha / lora_r
        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        result = self.pretrained_module(x)
        if self.lora_r > 0:
            after_dropout = self.lora_dropout(x)
            after_A = self.lora_A(after_dropout)
            after_B = self.lora_B(after_A)
            result += after_B * self.scaling
        return result

    def reset_parameters(self):
        # self.linear.reset_parameters()
        if self.lora_r > 0:
            self.lora_A.reset_parameters()
            self.lora_B.weight.data.zero_()

    @staticmethod
    def from_transformers_conv1d(
        original_layer,
        lora_r=0,
        lora_alpha=1.,
        lora_dropout=0.,
    ) -> "MYDPMergedLinear":
        lora_layer = MYDPMergedLinear(
            in_features=original_layer.weight.shape[0],
            out_features=original_layer.weight.shape[1],
            pretrained_module = original_layer,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        ).to(original_layer.weight.device)
        return lora_layer

def convert_gpt2_attention_to_lora(
    model: transformers.GPT2PreTrainedModel,
    lora_r=0,
    lora_alpha=1.,
    lora_dropout=0.,
) -> transformers.GPT2PreTrainedModel:
    if not isinstance(model, transformers.GPT2PreTrainedModel):
        raise TypeError("Requires a GPT2 model")

    if not hasattr(model, "h") and hasattr(model, "transformer"):
        transformer = model.transformer
    else:
        transformer = model

    for h_i in transformer.h:
        new_layer = MYDPMergedLinear.from_transformers_conv1d(
            original_layer=h_i.attn.c_attn,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        h_i.attn.c_attn = new_layer

    return model

def mutate_model(model: torch.nn.Module, lora_r=0, lora_alpha=1., lora_dropout=0.):
    for name, module in model.named_children():
        if name == "c_attn":
            new_layer = MYDPMergedLinear.from_transformers_conv1d(
                original_layer=module,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            setattr(model, name, new_layer)
        else:
            mutate_model(module, lora_r, lora_alpha, lora_dropout) # recursively call the function on the module


def mark_only_lora_as_trainable(model: torch.nn.Module) -> None:
    model.requires_grad_(True)
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False


class AdapterLayer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        bottleneck_size: int,
        bias = True
    ):
        super().__init__()

        self.sharif_llm_adapter = nn.Sequential(
            nn.Linear(emb_dim, bottleneck_size, bias=bias),
            nn.ReLU(),
            nn.Linear(bottleneck_size, emb_dim, bias=bias)
        )

    def forward(self, x: torch.Tensor):
        output = x + self.sharif_llm_adapter(x)
        return output

class FeedForwardAdapterWrapper(nn.Module):
    def __init__(
        self,
        original_module: GPT2MLP,
        bottleneck_size: int,
        emb_dim,
        bias = True
    ):

        super().__init__()

        assert isinstance(original_module, GPT2MLP)

        self.original_module = original_module
        self.adapter = AdapterLayer(emb_dim, bottleneck_size, bias=bias)

    def forward(self, x: torch.Tensor):
        output = self.original_module(x)
        output = self.adapter(output)
        return output
    
def mutate_model_recursive_adapter(model: nn.Module, bottleneck_size: int, emb_dim, bias=True):
    for name, module in model.named_children():
        if isinstance(module, GPT2MLP):
            feed_forward_with_adapter = FeedForwardAdapterWrapper(module, bottleneck_size, emb_dim, bias)
            setattr(model, name, feed_forward_with_adapter)
        else:
            mutate_model_recursive_adapter(module, bottleneck_size, emb_dim, bias) # recursively call the function on the module

def mutate_model_adapter(model: nn.Module, bottleneck_size: int, emb_dim, bias=True):
    if hasattr(model, '_mutated'):
        print("Model already contains adapter layers! \n Try reloading the model.")
        return

    mutate_model_recursive_adapter(model, bottleneck_size, emb_dim, bias)
    model._mutated = True
