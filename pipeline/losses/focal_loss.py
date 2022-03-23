from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(
        self,
        alpha: Optional[Union[Sequence, Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
        device: str = "cpu",
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            if not isinstance(alpha, Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.alpha = alpha.to(device=device)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
