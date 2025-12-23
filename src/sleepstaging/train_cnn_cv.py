# src/sleepstaging/train_cnn_cv.py
import argparse
import json
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

from .dataset import SleepEDFNPZDataset
from .labels import CLASSES
from .model_baseline import CNN1DBaseline
from .paths import MODELS, RESULTS, CFG
from .utils import set_seed
from .cv_splits import get_fold_subjects

def zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = x.mean(dim=-1, keepdim=True)
    s = x.std(dim=-1, keepdim=True)
    return (x - m) / (s + eps)

@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    for X, y in loader:
        X = X.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, dtype=torch.long,   non_blocking=True)
        X = zscore(X)
        logits = model(X)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y_pred.extend(pred)
        y_true.extend(y.detach().cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kap = cohen_kappa_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    return acc, f1, kap, cm

def run_fold(k: int, fold: int):
    set_seed(CFG["seed"])  # bendra sėkla (reproducibility)
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    train_subjects, val_subjects, test_subjects = get_fold_subjects(
        k=k, fold=fold, seed=CFG["seed"]
    )

    print(f"[CV] k={k} fold={fold}")
    print(f"[CV] TRAIN subjects ({len(train_subjects)}): {train_subjects[:5]} ...")
    print(f"[CV] VAL   subjects ({len(val_subjects)}):   {val_subjects}")
    print(f"[CV] TEST  subjects ({len(test_subjects)}):  {test_subjects[:5]} ...")

    train_ds = SleepEDFNPZDataset(subjects=train_subjects)
    val_ds   = SleepEDFNPZDataset(subjects=val_subjects)
    test_ds  = SleepEDFNPZDataset(subjects=test_subjects)

    n_classes = len(CLASSES)

    # WeightedRandomSampler kaip pas tave
    class_counts = np.bincount(train_ds.y.astype(int), minlength=n_classes).astype(np.float64)
    class_weights_for_sampler = class_counts.sum() / (class_counts + 1e-12)
    sample_weights = class_weights_for_sampler[train_ds.y.astype(int)]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=256, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== DEVICE: {device} ===")

    model = CNN1DBaseline(n_classes=n_classes).to(device=device, dtype=torch.float32)

    # LOSS: pas tave yra try su label_smoothing, bet ten buvo užkomentuota weights.
    # Kad nebūtų bugų, darom paprastai:
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_f1 = 0.0
    patience = 10
    no_improve = 0
    EPOCHS = 30

    t0 = time.time()
    print("=== START TRAIN (CV fold) ===")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for bi, (X, y) in enumerate(train_loader):
            X = X.to(device=device, dtype=torch.float32, non_blocking=True)
            y = y.to(device=device, dtype=torch.long,   non_blocking=True)
            X = zscore(X)

            opt.zero_grad(set_to_none=True)
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.detach().item())
            if bi % 10 == 0:
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    acc_b = (pred == y).float().mean().item()
                print(f"[F{fold}] [E{epoch:02d}] batch {bi:04d} | loss={loss.item():.4f} | acc={acc_b:.3f}")

        val_acc, val_f1, val_kap, _ = evaluate_full(model, val_loader, device)
        sched.step(val_f1)

        print(f"[F{fold}] [E{epoch:02d}] total_loss={total_loss:.3f} | VAL acc={val_acc:.3f} | F1={val_f1:.3f} | kappa={val_kap:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), MODELS / f"best_cnn1d_fold{fold}.pth")
            print(f"[SAVE] best fold={fold} (by VAL F1)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[STOP] Early stopping fold={fold}")
                break

    # TEST su best model
    best_path = MODELS / f"best_cnn1d_fold{fold}.pth"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state, strict=True)

    test_acc, test_f1, test_kap, test_cm = evaluate_full(model, test_loader, device)
    elapsed = time.time() - t0

    print(f"[F{fold}] TEST acc={test_acc:.3f} | macro-F1={test_f1:.3f} | kappa={test_kap:.3f}")
    print(f"[F{fold}] time={elapsed/60:.1f} min")

    out = {
        "k": k,
        "fold": fold,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_kappa": float(test_kap),
        "time_sec": float(elapsed),
        "confusion_matrix": test_cm.tolist(),
        "classes": CLASSES,
    }

    out_path = RESULTS / f"cv_cnn_fold{fold}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--fold", type=int, required=True)
    args = ap.parse_args()
    run_fold(k=args.k, fold=args.fold)

if __name__ == "__main__":
    main()

