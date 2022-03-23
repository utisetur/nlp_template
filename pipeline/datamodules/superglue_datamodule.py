from typing import List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from sklearn.utils import compute_class_weight
from transformers import DataCollatorWithPadding

from src.technical_utils import load_obj


class SuperGLUEDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = load_obj(cfg.datamodule.tokenizer).from_pretrained(
            cfg.datamodule.pretrained_tokenizer
        )
        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
        )
        self.dataset: Optional[Dataset] = None

    def prepare_data(self):
        # load datasets
        self.dataset = load_dataset(
            'super_glue', 
            self.cfg.datamodule.task_name.lower(),
            cache_dir=self.cfg.datamodule.data_path
        )

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset_class = load_obj(self.cfg.datamodule.class_name)

            self.train_dataset = dataset_class(
                data=self.dataset['train'],
                max_length=self.cfg.datamodule.max_length,
                tokenizer=self.tokenizer,
            )
            self.val_dataset = dataset_class(
                data=self.dataset['validation'],
                max_length=self.cfg.datamodule.max_length,
                tokenizer=self.tokenizer,
            )

        if stage == "test" or stage is None:
            dataset_class = load_obj(self.cfg.datamodule.class_name)
            self.test_dataset = dataset_class(
                data=self.dataset['test'],
                max_length=self.cfg.datamodule.max_length,
                tokenizer=self.tokenizer,
            )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.collator,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.collator,
            shuffle=False,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.collator,
            shuffle=False,
        )
        return test_loader
