import torch
from torch import nn


class LcnnLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels, **batch):
        """
        Args:
            logits (Tensor): (batch, 2) raw scores from the model.
            labels (Tensor): (batch,) ground-truth class indices.
        Returns:
            dict with the loss value under key "loss".
        """
        return {"loss": self.loss(logits, labels)}
