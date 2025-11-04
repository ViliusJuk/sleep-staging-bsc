import time, csv
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .paths import CFG, RESULTS, MODELS
from .utils import set_seed
from .labels import CLASSES
from .dataset_seq import SleepEDFSeqDataset
from .model_ma_bilstm import MA_CNN_BiLSTM

def make_loader(split, window_size=5, batch_size=16, weighted=True, shuffle=False):
    ds = SleepEDFSeqDataset(split=split, window_size=window_size)
    if weighted:
        counts = np.bincount([ds.y_list[fi][c] for fi,c in ds.windows], minlength=len(CLASSES)).astype(np.float64)
        probs = counts / counts.sum()
        class_w = (1.0 / (probs + 1e-9))
        sample_w = [class_w[ds.y_list[fi][c]] for fi,c in ds.windows]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0)
        return ds, loader, torch.tensor(class_w, dtype=torch.float32)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        return ds, loader, None

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for X, y in loader:
        X = X.to(device)
        pred = torch.argmax(model(X), dim=1).cpu().numpy()
        y_true.extend(y.numpy()); y_pred.extend(pred)
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0.0)
    kap = cohen_kappa_score(y_true, y_pred)
    return acc, f1, kap

def train():
    set_seed(CFG["seed"]); RESULTS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    window = 5
    train_ds, train_loader, class_w = make_loader("train", window_size=window, batch_size=64, weighted=True)
    val_ds,   val_loader,   _       = make_loader("val",   window_size=window, batch_size=128, weighted=False)

    model = MA_CNN_BiLSTM(n_classes=len(CLASSES)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w.to(device) if class_w is not None else None)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_f1, wait, patience, max_epochs = -1.0, 0, 10, 40
    log_path = RESULTS / "train_ma_bilstm_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","val_acc","val_macro_f1","val_kappa","lr"])

    for epoch in range(1, max_epochs+1):
        model.train()
        t0, run_loss, n_batches = time.time(), 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item(); n_batches += 1

        train_loss = run_loss / max(1, n_batches)
        val_acc, val_f1, val_kap = eval_epoch(model, val_loader, device)
        sched.step(val_f1)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_acc:.4f}", f"{val_f1:.4f}", f"{val_kap:.4f}", opt.param_groups[0]["lr"]])

        improved = val_f1 > best_f1 + 1e-4
        if improved:
            best_f1, wait = val_f1, 0
            torch.save(model.state_dict(), MODELS / "best_ma_cnn_bilstm.pth")
        else:
            wait += 1

        print(f"[{epoch:02d}/{max_epochs}] loss={train_loss:.4f} | val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_kappa={val_kap:.4f} | lr={opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.1f}s")
        if wait >= patience:
            print(f"Early stopping. Best val_macro_f1={best_f1:.4f}")
            break

if __name__ == "__main__":
    train()
