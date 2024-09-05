from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import copy
import sys
import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

def load_dataset(dataset_name, path, toy_example):
    dataset = load_from_disk(f"{path}saved_datasets/{dataset_name}")
    # toy example for develop
    if toy_example == 1:
        dataset["train"] = dataset["train"].select(range(1024))
        dataset["validation"] = dataset["validation"].select(range(512))
    return dataset


def load_dataloaders(dataset, dataset_name, batch_size, virtual_batch_size, tokenizer, seq_length, dp=1):
    data_collator = DataCollatorForData2TextLanguageModeling(tokenizer)
    if dataset_name == 'e2e_nlg':
        train_dataset = E2ETextDataset(tokenizer, 
                                       dataset["train"]["meaning_representation"], 
                                       dataset["train"]["human_reference"], 
                                       seq_length, 
                                       tokenizer.bos_token,
                                       tokenizer.eos_token,
                                       seq_length)
        validation_dataset = E2ETextDataset(tokenizer, 
                                            dataset["validation"]["meaning_representation"], 
                                            dataset["validation"]["human_reference"], 
                                            seq_length, 
                                            tokenizer.bos_token,
                                            tokenizer.eos_token,
                                            seq_length)

        train_data_size = len(dataset["train"])
        if dp == 1:
            sampler = WeightedRandomSampler([virtual_batch_size/train_data_size for _ in range(train_data_size)], num_samples=train_data_size, replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=virtual_batch_size, sampler=sampler, drop_last=True, collate_fn=data_collator)
        else:
            train_loader = DataLoader(train_dataset, batch_size=virtual_batch_size, collate_fn=data_collator)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, collate_fn=data_collator)
    elif dataset_name == 'dart':
        pass

    return train_loader, validation_loader


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

class E2ETextDataset(Dataset):

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        src_lines,
        tgt_lines,
        block_size: int,
        bos_tok: str,
        eos_tok: str,
        max_seq_len=sys.maxsize,
        max_examples=sys.maxsize,
        **_,
    ):
        src_lines = src_lines
        tgt_lines = tgt_lines

        edited_sents = []
        for src, tgt in zip(src_lines, tgt_lines):
            sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
            edited_sents.append(sent)

        # --- Filter out super long sentences ---
        new_src_lines, new_tgt_lines, new_edited_sents = [], [], []
        for src_line, tgt_line, edited_sent in zip(src_lines, tgt_lines, edited_sents):
            tokenized_edited_sent = tokenizer.tokenize(edited_sent)
            if len(tokenized_edited_sent) <= max_seq_len:
                new_src_lines.append(src_line)
                new_tgt_lines.append(tgt_line)
                new_edited_sents.append(edited_sent)
            del src_line, tgt_line, edited_sent
        src_lines, tgt_lines, edited_sents = new_src_lines, new_tgt_lines, new_edited_sents
        # ---------------------------------------

        # --- Truncate the dataset if necessary; this must be after the length filtering. ---
        src_lines = src_lines[:max_examples]
        tgt_lines = tgt_lines[:max_examples]
        edited_sents = edited_sents[:max_examples]
        # ---

        batch_encoding = tokenizer(
            edited_sents,
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            is_split_into_words=False,
        )

        self.examples = batch_encoding["input_ids"]
        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = []
        for ss in src_lines:
            ssl = [la.split(':')[0].strip() for la in ss.split('|')]
            ssl_lst.append(ssl)

        self.src_cat = tokenizer(
            ssl_lst,
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            is_split_into_words=True
        )['input_ids']

        self.src_sent = []
        self.tgt_sent = []

        # temp_src_len = 0
        # temp_tgt_len = 0
        # temp_count = 0

        separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
        for i, elem in enumerate(self.labels):
            sep_idx = elem.index(separator) + 1
            self.src_sent.append(self.examples[i][:sep_idx - 1])
            self.tgt_sent.append(self.examples[i][sep_idx - 1:])
            self.labels[i][:sep_idx] = [-100] * sep_idx  # Doesn't contribute to loss.
            # temp_src_len += sep_idx - 1
            # temp_tgt_len += len(elem) - (sep_idx - 1)
            # temp_count += 1

        # print('tgt_avg: ', temp_tgt_len / temp_count)
        # print('src_avg: ', temp_src_len / temp_count)
        # print('ratios: ', temp_src_len / temp_tgt_len)

        # print(self.labels[0])
        # print(self.examples[0])
        # print(edited_sents[0])
        # print(self.src_sent[0])
        # print(self.tgt_sent[0])
        # print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (
            torch.tensor(self.examples[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.long),
            torch.tensor(self.src_sent[i], dtype=torch.long),
            torch.tensor(self.tgt_sent[i], dtype=torch.long),
            torch.tensor(self.src_cat[i], dtype=torch.long),
        )
    


# InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset
and collate them into a batch, as a dictionary of Tensors.
"""
# DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])


@dataclass
class DataCollatorForData2TextLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    tokenizer: PreTrainedTokenizer
    mlm: bool = False
    format_mode: str = 'cat'
    mlm_probability: float = 0.15

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if isinstance(examples[0], (dict, BatchEncoding)):
            examples = [e["input_ids"] for e in examples]
        input_ids, labels, src, tgt, cate = zip(*examples)
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        else:
            if self.format_mode == 'cat':
                mode_input = 3
            elif self.format_mode == 'peek':
                mode_input = 1
            elif self.format_mode == 'nopeek':
                mode_input = 2
            elif self.format_mode == 'infix':
                mode_input = 4

            # mode_input = 1 # means that we take the input again.
            # mode_input = 2 # means that we do not peek at src again.
            # mode_input = 3 # means that we look at the categories, and see the input again.

            if mode_input == 1:
                # input, batch
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
                # tgt = self._tensorize_batch(tgt)
            elif mode_input == 2:
                # nopeek.
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)
                cate_batch, cate_attn = None, None
            elif mode_input == 3:
                batch = self._tensorize_batch(input_ids)
                labels = self._tensorize_batch(labels)
                src = self._tensorize_batch(cate)
                cate_batch, cate_attn = None, None
            elif mode_input == 4:
                batch = self._tensorize_batch(tgt)
                labels = batch.clone()
                src = self._tensorize_batch(src)

                cate_batch = self._tensorize_batch(cate)
                cate_attn = (cate_batch != self.tokenizer.pad_token_id)

            labels[labels == self.tokenizer.pad_token_id] = -100 # tgt
            src_attn = (src != self.tokenizer.pad_token_id) # src
            tgt_attn = (batch != self.tokenizer.pad_token_id) # tgt

            if cate_batch is None:
                return {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn':tgt_attn,
                        'src':src}
            else:
                return {"input_ids": batch, "labels": labels, 'src_attn': src_attn, 'tgt_attn': tgt_attn,
                        'src': src, "cate_batch":cate_batch, "cate_attn":cate_attn}

    def _tensorize_batch(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> torch.Tensor:
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )
            return pad_sequence(examples, batch_first=True, padding_value=self.tokenizer.pad_token_id)

    def mask_tokens(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
