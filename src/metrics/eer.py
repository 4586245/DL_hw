import torch

from calculate_eer import compute_eer
from src.metrics.base_metric import BaseMetric


class EerMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scores = []
        self._labels = []

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        scores = torch.softmax(logits, dim=-1)[:, 1]

        self._scores.append(scores.detach().cpu())
        self._labels.append(labels.detach().cpu())

        # EER is computed over the complete partition in compute().
        return 0.0

    def compute(self):
        """Compute EER for accumulated samples."""
        if not self._scores:
            return 0.0

        all_scores = torch.cat(self._scores)
        all_labels = torch.cat(self._labels)

        bona = all_scores[all_labels == 1].numpy()
        spoof = all_scores[all_labels == 0].numpy()

        if len(bona) == 0 or len(spoof) == 0:
            raise ValueError("EER requires both bonafide and spoof samples")

        eer, _ = compute_eer(bona, spoof)
        return eer * 100  # as in grading.py

    def reset(self):
        self._scores = []
        self._labels = []
