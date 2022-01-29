from typing import Tuple, Dict

import torch
from omegaconf import DictConfig
from torch import nn

from src.technical_utils import load_obj


class BiLSTM(nn.Module):
    """
    New model without nn.Embedding layer
    """

    def __init__(
        self,
        cfg: DictConfig,
        **kwargs: Dict,
    ):
        super(BiLSTM, self).__init__(**kwargs)
        self.cfg = cfg
        self.embedding_dropout = SpatialDropout(cfg.model.params.dropout)
        self.lstm = nn.LSTM(
            input_size=cfg.model.params.embeddings_dim,
            hidden_size=cfg.model.params.hidden_size // 2,
            num_layers=cfg.model.params.num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(self.cfg.model.params.hidden_size),
            # nn.Dropout(self.cfg.model.params.dropout),
            # nn.Linear(self.cfg.model.params.hidden_size, self.cfg.model.params.hidden_size),
            # nn.ReLU(),
            # nn.BatchNorm1d(self.cfg.model.params.hidden_size),
            # nn.Dropout(self.cfg.model.params.dropout),
            nn.Linear(self.cfg.model.params.hidden_size//2, self.cfg.model.params.n_classes),
        )
        # self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def _get_lstm_features(self, embeds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        LSTM forward

        Args:
            batch: batch with embeddings
        """
        embeds = self.embedding_dropout(embeds)
        lstm_feats, (h_n, c_n) = self.lstm(embeds)
        return lstm_feats, h_n, c_n

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward

        Args:
            batch: batch with embeddings
        """
        lstm_feats, h_n, c_n = self._get_lstm_features(embeds)
        # mean of layers
        logits = self.fc(torch.mean(c_n, dim=0))
        # the last hidden stage only
        # logits = self.fc(lstm_feats[:, -1, :])
        return logits

    def get_inference(self, embeds: torch.Tensor) -> torch.Tensor:
        return self(embeds)


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops a certain percentage of dimensions from each word vector in the training sample
    implementation: https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400
    explanation: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/76883
    """

    def __init__(self, p: float):
        super(SpatialDropout, self).__init__()
        self.spatial_dropout = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # convert to [batch, channels, time]
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, channels]
        return x
