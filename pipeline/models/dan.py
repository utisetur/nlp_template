import torch
from omegaconf import DictConfig
from torch import Tensor
import torch.nn as nn


class DAN(nn.Module):

    def __init__(self,
                 cfg: DictConfig,
                 *args,
                 **kwargs,
                 ):
        super(DAN, self).__init__()
        self.cfg = cfg
        self.model = nn.Sequential(
            nn.BatchNorm1d(cfg.model.params.embeddings_dim),
            nn.Dropout(cfg.model.params.dropout),
            nn.Linear(cfg.model.params.embeddings_dim, cfg.model.params.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(cfg.model.params.hidden_size),
            nn.Dropout(cfg.model.params.dropout),
            nn.Linear(cfg.model.params.hidden_size, cfg.model.params.n_classes)
        )

    def forward(self, batch):
        x = batch['input_ids']
        if self.cfg.model.params.aggregate == 'tfidf':
            token_weights = batch['tfidf']
            # TF-IDF weighted sum
            x = x * token_weights.unsqueeze(2)
            x = x.sum(axis=1)
        else:
            x = torch.mean(x, dim=1)
        x = self.model(x)
        return x
