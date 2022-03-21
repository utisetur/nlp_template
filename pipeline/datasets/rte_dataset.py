from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class RTEDataset(Dataset):
    """
    Custom PyTorch RTE train/val dataset class
    """

    def __init__(
        self, data: List, max_length: int, tokenizer: PreTrainedTokenizer, **kwarg: Dict
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels_map = {
            "not_entailment": 0,
            "entailment": 1,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sentence1, sentence2, label = self.data[idx]
        label = self.labels_map[label]

        encoding = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = (
            encoding.input_ids.squeeze(),
            encoding.attention_mask.squeeze(),
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }


class RTETestDataset(RTEDataset):
    """
    Custom PyTorch RTE test dataset class
    """

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # we need idx to make submission
        sentence1, sentence2, idx = self.data[idx]

        encoding = self.tokenizer.encode_plus(
            sentence1,
            sentence2,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = (
            encoding.input_ids.squeeze(),
            encoding.attention_mask.squeeze(),
        )

        return {
            "idx": idx,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
