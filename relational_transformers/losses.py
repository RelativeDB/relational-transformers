"""Task losses for custom PyTorch training loops."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class BinaryClassificationLoss(nn.Module):
    """Binary cross entropy over one logit per relational context."""

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits.reshape(-1), labels.float().reshape(-1))


class MulticlassClassificationLoss(nn.Module):
    """Cross entropy over mutually exclusive class logits."""

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(logits, labels.long())


class MultilabelClassificationLoss(nn.Module):
    """Binary cross entropy over independent label logits."""

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits, labels.float())


class RegressionLoss(nn.Module):
    """Huber loss for regression and forecasting targets."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, predictions: Tensor, labels: Tensor) -> Tensor:
        return F.huber_loss(predictions.reshape(-1), labels.float().reshape(-1), delta=self.delta)


class ListwiseRankingLoss(nn.Module):
    """Listwise cross entropy over candidate groups.

    Scores arrive as one logit per candidate; ``group_offsets`` splits them
    into groups. Relevance labels normalize into a target distribution per
    group, so every group needs at least one positive.
    """

    def forward(self, logits: Tensor, labels: Tensor, group_offsets) -> Tensor:
        scores = logits.reshape(-1)
        labels = labels.float().reshape(-1)
        n_groups = len(group_offsets) - 1
        total = scores.new_zeros(())
        for g in range(n_groups):
            s, e = int(group_offsets[g]), int(group_offsets[g + 1])
            log_p = F.log_softmax(scores[s:e], dim=0)
            target = labels[s:e] / labels[s:e].sum()
            total = total - (target * log_p).sum()
        return total / max(n_groups, 1)


LOSS_BY_PROBLEM_TYPE = {
    "binary": BinaryClassificationLoss,
    "multiclass": MulticlassClassificationLoss,
    "multilabel": MultilabelClassificationLoss,
    "regression": RegressionLoss,
    "forecasting": RegressionLoss,
}


def loss_for(problem_type: str) -> nn.Module:
    """Return the standard loss module for a supported problem type."""

    try:
        return LOSS_BY_PROBLEM_TYPE[problem_type]()
    except KeyError as exc:
        raise ValueError(f"unsupported problem_type {problem_type!r}") from exc
