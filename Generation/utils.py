import torch

def clean_hyperparameters(hyperparameters: dict):
    if hyperparameters["scheduler"] == 0:
        hyperparameters.pop("scheduler_type", None)
        hyperparameters.pop("scheduler_warmup_ratio", None)
        hyperparameters.pop("scheduler_warmup_steps", None)
        hyperparameters.pop("scheduler_step_size", None)
        hyperparameters.pop("scheduler_gamma", None)
    if hyperparameters["peft_mode"] != "lora":
        hyperparameters.pop("rank", None)
        hyperparameters.pop("alpha", None)
        hyperparameters.pop("drop_out", None)
    if hyperparameters["peft_mode"] != "adapter" and hyperparameters["peft_mode"] != "adapterbitfit":
        hyperparameters.pop("reduction_factor", None)
    if hyperparameters["dp"] == 0:
        hyperparameters.pop("epsilon", None)
        hyperparameters.pop("delta", None)
        hyperparameters.pop("clipping_mode", None)
        hyperparameters.pop("clipping_threshold", None)
    if hyperparameters["use_wandb"] == 0:
        hyperparameters.pop("wandb_project_name", None)
        hyperparameters.pop("use_wandb", None)
    if hyperparameters["two_step_training"] == 0:
        hyperparameters.pop("lr_two", None)
        hyperparameters.pop("virtual_batch_size_two", None)
        hyperparameters.pop("epochs_two", None)
        hyperparameters.pop("weight_decay_two", None)
    hyperparameters.pop("f", None)
    hyperparameters.pop("media_path", None)
    hyperparameters.pop("model_cache_path", None)
    return hyperparameters


def copy_model_weights(model1, model2):
    model1.eval()
    model2.eval()
    params1 = model1.parameters()
    params2 = model2.parameters()
    with torch.no_grad():
        for param1, param2 in zip(params1, params2):
            param2.data.copy_(param1.data)