from typing import Dict

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from transformers import DataCollatorWithPadding

from pipeline.datamodules.multitask_dataloader import MultitaskDataLoader
from src.text_utils import load_obj


class MultiTaskDataModule(pl.LightningDataModule):
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
        self.task_names = ["terra", "russe"]

    def prepare_data(self):
        # load datasets
        self.terra_train_data = pd.read_json(
            path_or_buf=self.cfg.datamodule.terra.data_path + "train.jsonl", lines=True
        ).set_index("idx")
        self.terra_val_data = pd.read_json(
            path_or_buf=self.cfg.datamodule.terra.data_path + "val.jsonl", lines=True
        ).set_index("idx")
        self.terra_test_data = pd.read_json(
            path_or_buf=self.cfg.datamodule.terra.data_path + "test.jsonl", lines=True
        ).set_index("idx")

        self.russe_train_data = pd.read_json(
            path_or_buf=self.cfg.datamodule.russe.data_path + "train.jsonl", lines=True
        ).set_index("idx")
        self.russe_val_data = pd.read_json(
            path_or_buf=self.cfg.datamodule.russe.data_path + "val.jsonl", lines=True
        ).set_index("idx")
        self.russe_test_data = pd.read_json(
            path_or_buf=self.cfg.datamodule.russe.data_path + "test.jsonl", lines=True
        ).set_index("idx")

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # train_data = prepare_data(self.train_data)
            # val_data = prepare_data(self.val_data)
            terra_dataset_class = load_obj(self.cfg.datamodule.terra.class_name)
            self.terra_train_dataset = terra_dataset_class(
                data=self.terra_train_data,
                tokenizer=self.tokenizer,
            )
            self.terra_valid_dataset = terra_dataset_class(
                data=self.terra_val_data,
                tokenizer=self.tokenizer,
            )

            russe_dataset_class = load_obj(self.cfg.datamodule.russe.class_name)
            self.russe_train_dataset = russe_dataset_class(
                data=self.russe_train_data,
                tokenizer=self.tokenizer,
            )
            self.russe_valid_dataset = russe_dataset_class(
                data=self.russe_val_data,
                tokenizer=self.tokenizer,
            )

        # if stage == "test":
        #     dataset_class = load_obj(self.cfg.inference.dataset_class)
        #     # test_data = prepare_data(self.test_data)
        #     self.test_dataset = dataset_class(
        #         data=self.test_data, tokenizer=self.tokenizer,
        #     )

    def train_dataloader(self):
        train_loader = MultitaskDataLoader(
            task_names=self.task_names,
            datasets=[self.terra_train_dataset, self.russe_train_dataset],
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.collator,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = MultitaskDataLoader(
            task_names=self.task_names,
            datasets=[self.terra_valid_dataset, self.russe_valid_dataset],
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.collator,
            shuffle=False,
        )
        return valid_loader

    # def test_dataloader(self):
    #     test_loader = torch.utils.data.DataLoader(
    #         self.test_dataset,
    #         batch_size=self.cfg.datamodule.batch_size,
    #         num_workers=self.cfg.datamodule.num_workers,
    #         pin_memory=self.cfg.datamodule.pin_memory,
    #         collate_fn=self.collator,
    #         shuffle=False,
    #     )
    #     return test_loader
