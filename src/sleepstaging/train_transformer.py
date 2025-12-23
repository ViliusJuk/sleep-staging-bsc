import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from .dataset import SleepEDFNPZDataset
from .labels import CLASSES
from .model_transformer import SleepTransformer
from .paths import MODELS, RESULTS, CFG
from .utils import set_seed


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    return (x - m) / (s + eps)


def make_weighted_sampler(ds):
    counts = np.bincount(ds.y.astype(int), minlength=len(CLASSES)).astype(np.float64)
    freq = counts / counts.sum()
    inv = 1.0 / (freq + 1e-9)
    sample_weights = inv[ds.y.astype(int)]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    for X, y in loader:
        X = zscore(X.to(device, dtype=torch.float32, non_blocking=True))
        y = y.to(device, dtype=torch.long, non_blocking=True)
        logits = model(X)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y_pred.extend(pred)
        y_true.extend(y.detach().cpu().numpy())
    if not y_true:
        return 0.0, 0.0, 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    return acc, f1, kappa


def train():
    set_seed(CFG["seed"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # --- DATA ---
    train_ds = SleepEDFNPZDataset(split="train")
    val_ds   = SleepEDFNPZDataset(split="val")

    sampler = make_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=96, sampler=sampler, num_workers=0, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"[INFO] Transformer | TRAIN={len(train_ds)} | VAL={len(val_ds)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DEVICE: {device} ===")

    # --- MODEL ---
    model = SleepTransformer(
        n_classes=len(CLASSES),
        d_model=128,
        n_heads=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
    )
    model = model.to(device=device, dtype=torch.float32)
    print("[DBG] model dtype:", next(model.parameters()).dtype)

    # --- LOSS / OPT / SCHED ---
    # galima pridėti class weights, jeigu reikės, bet pradžiai pakanka samplerio
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3
    )

    best_f1 = 0.0
    no_improve = 0
    patience = 10
    EPOCHS = 40

    print("=== START TRAIN (Transformer) ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for bi, (X, y) in enumerate(train_loader):
            X = zscore(X.to(device, dtype=torch.float32, non_blocking=True))
            y = y.to(device, dtype=torch.long, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.detach().item())
            if bi % 10 == 0:
                with torch.no_grad():
                    acc_batch = (logits.argmax(1) == y).float().mean().item()
                print(
                    f"[E{epoch:02d}] b{bi:04d} | loss={loss.item():.4f} | acc={acc_batch:.3f}"
                )

        acc, f1, kappa = evaluate(model, val_loader, device)
        sched.step(f1)
        print(
            f"[E{epoch:02d}] total_loss={total_loss:.3f} | "
            f"VAL acc={acc:.3f} | F1={f1:.3f} | kappa={kappa:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            torch.save(model.state_dict(), MODELS / "best_transformer.pth")
            print("[SAVE] New best Transformer (by F1)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("[STOP] Early stopping (no improvement)")
                break

    torch.save(model.state_dict(), MODELS / "last_transformer.pth")
    print("=== TRAIN DONE (Transformer) ===")


if __name__ == "__main__":
    train()

