import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.optim import AdamW

from .datasets.sleepedf import build_dataloaders, get_class_counts
from .losses.class_balanced_ce import ClassBalancedCELoss
from .schedulers.cosine import build_cosine_scheduler
from .trainers.engine import train_one_epoch, evaluate as validate

from sklearn.metrics import f1_score, cohen_kappa_score


def metrics_fn(y_true, y_pred):
    # skaičiuojam tik per klases, kurios egzistuoja validation set'e
    labels = np.unique(y_true)
    macro_f1 = f1_score(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    return {"macro_f1": macro_f1, "kappa": kappa}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug_overfit", type=int, default=0,
                    help="If >0, train only on first N samples for quick overfit test")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument(
        "--model_cfg",
        type=json.loads,
        default='{"d_model":128,"nhead":4,"num_layers":4,"dim_feedforward":256,"dropout":0.1,"num_classes":5}'
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== DEVICE: {device} ===")

    # data
    train_loader, val_loader, info = build_dataloaders(
        args.data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True
    )
    print(f"[INFO] SEQ_LEN={info['seq_len']} | TRAIN={info['n_train']} | VAL={info['n_val']}")

        # DEBUG QUICK OVERFIT CHECK (įdėta po build_dataloaders ir info print)
    if getattr(args, "debug_overfit", 0) > 0:
        from torch.utils.data import Subset, DataLoader

        n_tr = min(len(train_loader.dataset), args.debug_overfit)
        n_va = min(len(val_loader.dataset),   max(1, args.debug_overfit // 4))

        train_subset = Subset(train_loader.dataset, list(range(n_tr)))
        val_subset   = Subset(val_loader.dataset,   list(range(n_va)))

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        print(f"[DEBUG] Overfitting mode ON: train={n_tr}, val={n_va}")
        # šitam režime leidžiam daugiau epochų, bet Warmup paliekam nedidelį
        args.epochs = max(args.epochs, 200)
        args.warmup_epochs = min(args.warmup_epochs, 5)



    # compute class counts
    class_counts = get_class_counts(data_root=args.data)
    print(f"[INFO] Train class counts: {class_counts}")

    # adjust num_classes dynamically
    n_classes = len(class_counts)
    args.model_cfg["num_classes"] = n_classes

    # build model
    from .model_transformer import SleepTransformer
    model = SleepTransformer(**args.model_cfg).to(device)

    # loss (CB CE)
    loss_fn = ClassBalancedCELoss(samples_per_cls=class_counts, beta=0.9999).to(device)

    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # scheduler (cosine + warmup)
    scheduler = build_cosine_scheduler(
        optimizer,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr
    )

    best_f1 = -1
    no_improve_epochs = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            device, max_grad_norm=args.max_grad_norm, use_amp=args.amp
        )

        val_loss, val_acc, extra = validate(
            model, val_loader, loss_fn, device, metric_fn=metrics_fn
        )

        f1 = extra.get("macro_f1", 0.0)
        kappa = extra.get("kappa", 0.0)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[E{epoch:02d}] loss={val_loss:.3f} | acc={val_acc:.3f} | "
            f"F1={f1:.3f} | κ={kappa:.3f} | LR={lr:.6f}"
        )

        scheduler.step()

        if f1 > best_f1:
            best_f1 = f1
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_transformer.pt")
            print("[SAVE] ✅ New best model")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.patience:
            print("[STOP] Early stopping (no improvement)")
            break

    print("=== TRAIN DONE ===")


if __name__ == "__main__":
    main()

