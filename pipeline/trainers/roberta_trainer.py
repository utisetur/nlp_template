import glob
from functools import lru_cache
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn

from src.technical_utils import load_obj


class RoBERTaFineTuner(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        *args: List,
        **kwargs: Dict,
    ):
        super(RoBERTaFineTuner, self).__init__()
        self.cfg = cfg
        self.model = load_obj(cfg.model.class_name)(cfg)
        if cfg.model.checkpoint:
            print("load from the checkpoint...")
            model_names = glob.glob(
                f"../../outputs/{cfg.model.checkpoint}/saved_models/*"
            )
            best_model = [name for name in model_names if "best" in name][0]
            checkpoint = torch.load(best_model)
            self.model.load_state_dict(checkpoint)
        if cfg.model.freeze_params:
            print("freeze backbone...")
            self.freeze_params()

        self.criterion = load_obj(cfg.loss.class_name)(**cfg.loss.params)
        self.metrics = nn.ModuleDict(
            {
                self.cfg.metrics.metric.metric_name: load_obj(
                    self.cfg.metrics.metric.class_name
                )(**cfg.metrics.metric.params).to(self.cfg.general.device)
            }
        )
        if "other_metrics" in self.cfg.metrics.keys():
            for metric in self.cfg.metrics.other_metrics:
                self.metrics[metric.metric_name] = load_obj(metric.class_name)(
                    **metric.params
                ).to(self.cfg.general.device)

    def freeze_params(self):
        for param in self.model.model.parameters():
            param.requires_grad = False

    def forward(self, batch):
        logits = self.model(**batch)
        return logits

    @lru_cache()
    def total_steps(self):
        return (
            self.cfg.datamodule.train_dataloader_size
            // self.cfg.trainer.accumulate_grad_batches
        ) * self.cfg.trainer.max_epochs

    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.optimizer.class_name)(
            self.model.parameters(), **self.cfg.optimizer.params
        )

        if "num_warmup_steps" in self.cfg.scheduler.params:
            if not self.cfg.scheduler.params.num_warmup_steps:
                num_warmup_steps = int(
                    self.total_steps() * self.cfg.scheduler.warmup_steps_fraction
                )
            else:
                num_warmup_steps = self.cfg.scheduler.backbone.params.num_warmup_steps
            scheduler = load_obj(self.cfg.scheduler.class_name)(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps(),
            )
        else:
            scheduler = load_obj(self.cfg.scheduler.class_name)(
                optimizer,
            )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": self.cfg.scheduler.step,
                    "monitor": self.cfg.scheduler.monitor,
                }
            ],
        )

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        logits = self(batch)
        loss = self.criterion(logits, labels)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        probs = torch.softmax(logits, dim=1)
        for metric_name, metric in self.metrics.items():
            score = metric(probs, labels)
            self.log(
                f"train_{metric_name}",
                score,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        logits = self(batch)
        loss = self.criterion(logits, labels)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        probs = torch.softmax(logits, dim=1)
        for metric_name, metric in self.metrics.items():
            score = metric(probs, labels)
            self.log(
                f"val_{metric_name}",
                score,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, batch_idx):
        labels = batch["labels"]
        logits = self(batch)
        loss = self.criterion(logits, labels)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        probs = torch.softmax(logits, dim=1)
        for metric_name, metric in self.metrics.items():
            score = metric(probs, labels)
            self.log(
                f"test_{metric_name}",
                score,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
