from __future__ import annotations

import torch
from torch import nn
from torch.amp import autocast, GradScaler

def _unpack_batch(batch):
    if isinstance(batch, dict):
        x, y = batch["x"], batch["y"]
    else:
        x, y = batch
    return x, y

@torch.no_grad()
def evaluate(model: nn.Module, loader, loss_fn, device: torch.device, use_amp: bool = False, metric_fn=None):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_pred = []
    all_true = []

    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        n += int(y.size(0))

        all_pred.append(pred.detach().cpu())
        all_true.append(y.detach().cpu())

    avg_loss = total_loss / max(1, n)
    acc = correct / max(1, n)

    extra = {}
    if metric_fn is not None:
        y_true = torch.cat(all_true).numpy()
        y_pred = torch.cat(all_pred).numpy()
        # metric_fn turėtų grąžinti dict (pvz., {"f1":..., "kappa":...})
        try:
            extra = metric_fn(y_true, y_pred) or {}
        except Exception as e:
            extra = {"metric_error": str(e)}

    return avg_loss, acc, extra

