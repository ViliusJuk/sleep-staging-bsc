import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from .dataset import SleepEDFNPZDataset
from .labels import CLASSES
from .model_baseline import CNN1DBaseline
from .paths import MODELS, RESULTS, CFG
from .utils import set_seed


def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-epoch z-score normalizacija: (E, 1, T) per T ašį."""
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    return (x - m) / (s + eps)


def compute_epoch_class_weights(ds, n_classes: int, smoothing: float = 1.0) -> torch.Tensor:
    """
    Stabilūs klasės svoriai su Laplace smoothing (+1) ir normalizacija į mean=1.
    """
    counts = np.bincount(ds.y.astype(int), minlength=n_classes).astype(np.float64)
    counts += float(smoothing)
    freq = counts / counts.sum()
    w = 1.0 / (freq + 1e-12)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []

    for X, y in loader:
        X = X.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.long, non_blocking=True)
        X = zscore(X)

        logits = model(X)  # tikimės RAW LOGITS
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
    val_ds = SleepEDFNPZDataset(split="val")

    # Subalansuotas mėginių ėmimas per batch'us
    n_classes = len(CLASSES)
    class_counts = np.bincount(train_ds.y.astype(int), minlength=n_classes).astype(np.float64)
    class_weights_for_sampler = class_counts.sum() / (class_counts + 1e-12)
    sample_weights = class_weights_for_sampler[train_ds.y.astype(int)]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=256, sampler=sampler, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"[INFO] Epoch-level CNN | TRAIN={len(train_ds)} | VAL={len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DEVICE: {device} ===")

    # --- MODEL ---
    model = CNN1DBaseline(n_classes=n_classes).to(device=device, dtype=torch.float32)
    print("[DBG] model dtype:", next(model.parameters()).dtype)

    # Greitas sanity check ant pirmo batch'o
    X0, y0 = next(iter(train_loader))
    print("[DBG] first batch dtypes:", X0.dtype, y0.dtype, "| shape:", X0.shape)
    del X0, y0

    # --- LOSS ---
    # Pastaba: jei naudoji label_smoothing, class weights dažnai nebūtini.
    # Bet jei torch versija nepalaiko label_smoothing, turim fallback su weights.
    weights = compute_epoch_class_weights(train_ds, n_classes).to(device)
    print("[INFO] class weights (CE):", weights.detach().cpu().numpy())

    try:
        # naujesnė PyTorch versija palaiko label_smoothing
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    except TypeError:
        # fallback senesnei PyTorch versijai (be label_smoothing)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

    # --- OPT/SCHED ---
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3
    )

    best_f1 = 0.0
    patience = 10
    no_improve = 0
    EPOCHS = 30

    print("=== START TRAIN ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for bi, (X, y) in enumerate(train_loader):
            X = X.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.long, non_blocking=True)
            X = zscore(X)

            opt.zero_grad(set_to_none=True)
            logits = model(X)  # RAW LOGITS
            loss = loss_fn(logits, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.detach().item())
            if bi % 10 == 0:
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    acc_b = (pred == y).float().mean().item()
                print(f"[E{epoch:02d}] batch {bi:04d} | loss={loss.detach().item():.4f} | acc={acc_b:.3f}")

        acc, f1, kappa = evaluate(model, val_loader, device)
        sched.step(f1)

        print(
            f"[E{epoch:02d}] total_loss={total_loss:.3f} | "
            f"VAL acc={acc:.3f} | F1={f1:.3f} | kappa={kappa:.3f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            torch.save(model.state_dict(), MODELS / "best_cnn1d.pth")
            print("[SAVE] New best CNN1D (by F1)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("[STOP] Early stopping (no improvement)")
                break

    torch.save(model.state_dict(), MODELS / "last_cnn1d.pth")
    print("=== TRAIN DONE ===")


if __name__ == "__main__":
    train()

