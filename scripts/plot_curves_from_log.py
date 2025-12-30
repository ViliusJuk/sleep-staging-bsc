import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def parse_log(path: Path):
    """
    Ištraukia:
      - train total_loss iš eilučių: [E##] total_loss=...
      - val acc / F1 / kappa iš eilučių: [E##] ... VAL acc=... | F1=... | kappa=...
    """
    text = path.read_text(errors="ignore")

    # pvz: [E11] total_loss=475.808 | VAL acc=0.881 | F1=0.701 | kappa=0.769
    pat = re.compile(
        r"\[E(\d+)\]\s+total_loss=([0-9.]+)\s+\|\s+VAL acc=([0-9.]+)\s+\|\s+F1=([0-9.]+)\s+\|\s+kappa=([0-9.]+)"
    )

    epochs, loss, acc, f1, kappa = [], [], [], [], []
    for m in pat.finditer(text):
        epochs.append(int(m.group(1)))
        loss.append(float(m.group(2)))
        acc.append(float(m.group(3)))
        f1.append(float(m.group(4)))
        kappa.append(float(m.group(5)))

    return epochs, loss, acc, f1, kappa

def save_loss_curve(epochs, loss, out_path: Path, title: str):
    fig = plt.figure(figsize=(6.5, 4.5))
    plt.plot(epochs, loss)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_val_metrics(epochs, acc, f1, kappa, out_path: Path, title: str):
    fig = plt.figure(figsize=(6.5, 4.5))
    plt.plot(epochs, acc, label="VAL acc")
    plt.plot(epochs, f1, label="VAL macro-F1")
    plt.plot(epochs, kappa, label="VAL kappa")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="pvz: logs/bilstm_196907.out")
    ap.add_argument("--tag", required=True, help="pvz: bilstm arba cnn arba transformer")
    ap.add_argument("--outdir", default="figures", help="kur saugoti PNG")
    args = ap.parse_args()

    log_path = Path(args.log)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    epochs, loss, acc, f1, kappa = parse_log(log_path)
    if len(epochs) == 0:
        raise SystemExit(
            "Neradau eilučių su patternu '[E##] total_loss=... | VAL acc=... | F1=... | kappa=...'\n"
            "Patikrink, kaip tiksliai atrodo tavo log'o 'summary' eilutės."
        )

    save_loss_curve(epochs, loss, outdir / f"{args.tag}_train_loss.png", f"{args.tag.upper()} training curve: loss per epoch")
    save_val_metrics(epochs, acc, f1, kappa, outdir / f"{args.tag}_val_metrics.png", f"{args.tag.upper()} validation metrics per epoch")

    print("Saved:")
    print(outdir / f"{args.tag}_train_loss.png")
    print(outdir / f"{args.tag}_val_metrics.png")

if __name__ == "__main__":
    main()

