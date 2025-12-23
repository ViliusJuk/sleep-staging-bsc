# src/sleepstaging/losses/class_balanced_ce.py
import math
import torch
import torch.nn as nn

class ClassBalancedCELoss(nn.Module):
    """
    Cui et al. Class-Balanced Loss for imbalanced data.
    samples_per_cls: list/tuple of counts per class from TRAIN split.
    """
    def __init__(self, samples_per_cls, beta=0.9999, label_smoothing=0.0):
        super().__init__()
        effective_num = [1.0 - math.pow(beta, n) for n in samples_per_cls]
        weights = [(1.0 - beta) / en if en > 0 else 0.0 for en in effective_num]
        w = torch.tensor(weights, dtype=torch.float)
        w = w / w.sum() * len(w)  # normalize around 1.0
        self.register_buffer("weights", w)
        self.ce = nn.CrossEntropyLoss(weight=self.weights, label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        return self.ce(logits, targets)

