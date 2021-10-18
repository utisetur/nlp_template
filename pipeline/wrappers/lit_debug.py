from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.technical_utils import load_obj


class DebugWrapper(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(DebugWrapper, self).__init__()
        self.cfg = cfg
        self.model = load_obj(cfg.model.class_name)(
            sequence_length=self.cfg.model.params.sequence_length,
            n_cell_types=self.cfg.model.params.n_cell_types,
            sequence_embedding_length=self.cfg.model.params.sequence_embedding_length,
            cell_type_embedding_length=self.cfg.model.params.cell_type_embedding_length,
            final_embedding_length=self.cfg.model.params.final_embedding_length,
            n_genomic_features=self.cfg.model.params.n_genomic_features,
        )
        self.criterion = load_obj(cfg.loss.class_name)()
        self.metrics = [
            {
                'metric': load_obj(self.cfg.metric.metric.class_name)(**cfg.metric.metric.params),
                'metric_name': self.cfg.metric.metric.metric_name,
            }
        ]
        if 'other_metrics' in self.cfg.metric.keys():
            for metric in self.cfg.metric.other_metrics:
                self.metrics.append(
                    {
                        'metric': load_obj(metric.class_name)(**metric.params).to(self.cfg.general.device),
                        'metric_name': metric.metric_name,
                    }
                )

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return (
            [optimizer],
            [{'scheduler': scheduler, 'interval': self.cfg.scheduler.step, 'monitor': self.cfg.scheduler.monitor}],
        )

    def training_step(self, batch, batch_idx):
        sequence_batch, targets, target_mask = batch
        predictions = self.model(sequence_batch)
        # predictions =
        loss = self.criterion(predictions, targets)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # for metric in self.metrics:
        #     score = metric['metric'](predictions, targets)
        #     self.log(f"train_{metric['metric_name']}", score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequence_batch, targets, target_mask = batch
        predictions = self.model(sequence_batch)
        loss = self.criterion(predictions, targets)

        self.log('valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # for metric in self.metrics:
        #     score = metric['metric'](predictions, targets)
        #     self.log(f"{metric['metric_name']}", score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
