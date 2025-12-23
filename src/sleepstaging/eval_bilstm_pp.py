import torch, numpy as np, itertools
from sklearn.metrics import classification_report, f1_score, confusion_matrix, cohen_kappa_score, accuracy_score
from torch.utils.data import DataLoader

from .dataset import SleepEDFNPZDataset
from .labels import CLASSES
from .model_bilstm import CNNBiLSTM
from .paths import MODELS

def zscore(x, eps=1e-6):
    m=x.mean(dim=-1,keepdim=True); s=x.std(dim=-1,keepdim=True); return (x-m)/(s+eps)

def smooth_mode(y_pred: np.ndarray, k: int) -> np.ndarray:
    y = np.asarray(y_pred); n=len(y); h=k//2
    out = np.empty(n, dtype=y.dtype)
    for i in range(n):
        a=max(0,i-h); b=min(n,i+h+1)
        w=y[a:b]; out[i]=np.bincount(w, minlength=len(CLASSES)).argmax()
    return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== DEVICE:", device, "===")

    ds = SleepEDFNPZDataset(split="val")
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    model = CNNBiLSTM(n_classes=len(CLASSES)).to(device, dtype=torch.float32)
    state = torch.load(MODELS / "best_bilstm.pth", map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    logits_list, y_true = [], []
    with torch.no_grad():
        for X, y in loader:
            X = zscore(X.to(device, dtype=torch.float32))
            L = model(X).cpu().numpy()
            logits_list.append(L)
            y_true.extend(y.numpy())

    logits = np.vstack(logits_list)
    y_true = np.array(y_true)

    # RAW
    y_raw = logits.argmax(1)
    acc_raw = accuracy_score(y_true, y_raw)
    f1_raw  = f1_score(y_true, y_raw, average="macro", zero_division=0)
    kap_raw = cohen_kappa_score(y_true, y_raw)
    print(f"RAW: acc={acc_raw:.3f} | macro-F1={f1_raw:.3f} | kappa={kap_raw:.3f}")

    # BIAS grid tik N1 ir N3 (pagal ankstesnę patirtį)
    best = (None, -1e9)
    for b1, b3 in itertools.product(np.linspace(0.0,-1.0,11), np.linspace(0.0,-1.0,11)):
        bias = np.array([0, b1, 0, b3, 0], dtype=np.float32)
        y_b = (logits + bias).argmax(1)
        f1 = f1_score(y_true, y_b, average="macro", zero_division=0)
        if f1 > best[1]:
            best = ((b1, b3), f1)
    (b1, b3), f1_best = best
    print(f"Best bias (N1,N3)=({b1:.2f},{b3:.2f}) | macro-F1={f1_best:.3f}")

    # BIAS + SMOOTH
    y_bias = (logits + np.array([0,b1,0,b3,0], dtype=np.float32)).argmax(1)
    y_bs   = smooth_mode(y_bias, k=5)

    acc = accuracy_score(y_true, y_bs)
    f1  = f1_score(y_true, y_bs, average="macro", zero_division=0)
    kap = cohen_kappa_score(y_true, y_bs)
    print(f"BIAS+SMOOTH: acc={acc:.3f} | macro-F1={f1:.3f} | kappa={kap:.3f}")

    print("\nPer-klasinė metrika (BIAS+SMOOTH):")
    print(classification_report(y_true, y_bs, target_names=CLASSES, digits=3))
    print("Confusion matrix (BIAS+SMOOTH):")
    print(confusion_matrix(y_true, y_bs))

if __name__ == "__main__":
    main()

