import torch
from torch import nn


class Seq2SeqCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, pad_id: int):
        super().__init__(ignore_index=pad_id)

    def normalization_weight(self, labels: torch.Tensor, **batch) -> int:
        """Count on CPU before transfer, to weight accumulated microbatches by tokens."""
        return int((labels != self.ignore_index).sum().item())

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, **batch
    ) -> dict[str, torch.Tensor]:
        if logits.ndim != 3 or labels.ndim != 2:
            raise ValueError("Expected logits [B, T, V] and labels [B, T]")
        if logits.shape[:2] != labels.shape:
            raise ValueError("Logits and labels must have matching batch and length")

        loss = super().forward(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )
        return {"loss": loss}
