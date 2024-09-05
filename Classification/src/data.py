from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler


TASK_TO_KEYS = {
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "sst2": ("sentence", None),
}

def prepare_data(cfg, tokenizer):
    dataset = load_from_disk(f"{cfg.media_path}saved_datasets/{cfg.dataset}")
    sentence1_key, sentence2_key = TASK_TO_KEYS[cfg.dataset]

    if cfg.toy_example:
        dataset["train"] = dataset["train"].select(range(1024))

    def tokenize(batch):
        args = ((batch[sentence1_key],) if sentence2_key is None else (batch[sentence1_key], batch[sentence2_key]))
        return tokenizer(*args, padding="max_length", truncation=True, max_length=cfg.max_length)

    dataset = dataset.map(tokenize, batched=True, batch_size=len(dataset))
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    cfg.train_data_size = len(dataset['train'])

    sampler = WeightedRandomSampler([cfg.virtual_batch_size/cfg.train_data_size for _ in range(cfg.train_data_size)], num_samples=cfg.train_data_size, replacement=True)
    train_loader = DataLoader(dataset['train'], batch_size=cfg.virtual_batch_size, sampler=sampler, drop_last=True)

    validation_loader_one = None
    validation_loader_two = None
    if cfg.dataset == "mnli":
        if cfg.toy_example:
            dataset["validation_matched"] = dataset["validation_matched"].select(range(100))
            dataset["validation_mismatched"] = dataset["validation_mismatched"].select(range(100))
        validation_loader_one = DataLoader(dataset['validation_matched'], batch_size=cfg.batch_size)
        validation_loader_two = DataLoader(dataset['validation_mismatched'], batch_size=cfg.batch_size)
    else:
        if cfg.toy_example:
            dataset["validation"] = dataset["validation"].select(range(100))
        validation_loader_one = DataLoader(dataset['validation'], batch_size=cfg.batch_size)

    return train_loader, validation_loader_one, validation_loader_two
