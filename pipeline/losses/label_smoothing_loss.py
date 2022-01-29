from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(
        self,
        smoothing: float = 0.1,
        use_kl_div: bool = False,
        ignore_index: int = 0,
        reduce: bool = True,
    ):
        super().__init__()

        assert 0 < smoothing < 1

        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.use_kl_div = use_kl_div
        self.reduce = reduce

    def smooth_one_hot(self, true_labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:

        confidence = 1.0 - self.smoothing

        with torch.no_grad():
            true_dist = torch.empty(size=(true_labels.size(0), num_classes,), device=true_labels.device,)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(
                1, true_labels.data.unsqueeze(1), confidence,
            )

        return true_dist

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        :param logits: [batch_size, num_classes]
        :param targets: [batch_size]
        :param mask: [batch_size] True if need
        :return: scalar
        """

        # logits = F.log_softmax(logits, dim=-1, dtype=torch.float32)

        targets_smoothed_dist = self.smooth_one_hot(targets, num_classes=2)

        if self.use_kl_div:
            loss = - F.kl_div(logits, targets_smoothed_dist, reduction="none",).sum(dim=-1)
        else:
            loss = torch.sum(targets_smoothed_dist * logits, dim=-1,)

        if self.reduce:
            loss = loss.mean()

        return loss



# class LabelSmoothingLoss(nn.Module):
#     """
#     With label smoothing,
#     KL-divergence between q_{smoothed ground truth prob.}(w)
#     and p_{prob. computed by model}(w) is minimized.
#     """
#     def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
#         assert 0.0 < label_smoothing <= 1.0
#         self.ignore_index = ignore_index
#         super(LabelSmoothingLoss, self).__init__()
#
#         smoothing_value = label_smoothing / (tgt_vocab_size - 2)
#         one_hot = torch.full((tgt_vocab_size,), smoothing_value)
#         one_hot[self.ignore_index] = 0
#         self.register_buffer('one_hot', one_hot.unsqueeze(0))
#
#         self.confidence = 1.0 - label_smoothing
#
#     def forward(self, output, target):
#         """
#         output (FloatTensor): batch_size x n_classes
#         target (LongTensor): batch_size
#         """
#         model_prob = self.one_hot.repeat(target.size(0), 1)
#         model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
#         model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
#
#         return F.kl_div(output, model_prob, reduction='sum')
