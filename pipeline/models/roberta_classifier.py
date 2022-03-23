import torch
from omegaconf import DictConfig
from torch import nn
from transformers import RobertaModel


class RoBERTaClassifier(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.model = RobertaModel.from_pretrained(cfg.model.pretrained_model)

        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(self.model.config.hidden_size),
            nn.Dropout(p=self.cfg.model.params.dropout),
            nn.Linear(
                self.model.pooler.dense.weight.shape[0], self.cfg.model.params.n_classes
            ),
        )

    def forward(self, input_ids, attention_mask=None, *args, **kwargs):
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = model_output.pooler_output
        logits = self.classifier(embeddings)
        return logits
