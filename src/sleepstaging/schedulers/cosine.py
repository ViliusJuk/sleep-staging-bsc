# src/sleepstaging/schedulers/cosine.py
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

def build_cosine_scheduler(optimizer, epochs, warmup_epochs=0, min_lr=1e-6):
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr)
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    return cosine

