from typing import Dict, List

from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import datasets


class RTEDataset(Dataset):
    """
    Custom PyTorch RTE train/val dataset class
    """

    def __init__(
        self, data: datasets.arrow_dataset.Dataset, max_length: int, tokenizer: PreTrainedTokenizer, **kwarg: Dict
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels_map = {
            0: "not_entailment",
            1: "entailment",
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.data[idx]
        label = sample['label']

        encoding = self.tokenizer.encode_plus(
            sample['premise'],
            sample['hypothesis'],
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
            "idx": sample['idx'],
        }
