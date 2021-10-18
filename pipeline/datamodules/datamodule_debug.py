from typing import Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.technical_utils import load_obj


class DebugDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = load_obj(self.cfg.datamodule.class_name)(cfg=self.cfg)

        self.val_dataset = load_obj(self.cfg.datamodule.class_name)(cfg=self.cfg)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.datamodule.batch_size,
        )
        print('train loader len:', len(train_loader))
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.datamodule.batch_size,
        )
        print('val loader len:', len(valid_loader))
        return valid_loader

    def test_dataloader(self):
        return None
