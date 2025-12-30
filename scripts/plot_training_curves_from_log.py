import re
from pathlib import Path
import matplotlib.pyplot as plt

LOG = Path("logs/bilstm_196907.out")  # pakeisk į savo
pat = re.compile(r"\[E(\d+)\]\s+total_loss=([0-9.]+)\s+\|\s+VAL acc=([0-9.]+)\s+\|\s+F1=([0-9.]+)\s+\|\s+kappa=([0-9.]+)")

epochs, loss, acc, f1, kappa = [], [], [], [], []

for line in LOG.read_text().splitlines():
    m = pat.search(line)
    if m:
        epochs.append(int(m.group(1)))
        loss.append(float(m.group(2)))
        acc.append(float(m.group(3)))
        f1.append(float(m.group(4)))
        kappa.append(float(m.group(5)))

if not epochs:
    raise SystemExit("Nieko neradau log'e. Patikrink LOG kelią ir regex.")

plt.figure()
plt.plot(epochs, loss)
plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.title("Training curve: loss per epoch")
plt.tight_layout()
plt.savefig("figures/train_loss.png", dpi=200)

plt.figure()
plt.plot(epochs, acc, label="VAL acc")
plt.plot(epochs, f1, label="VAL macro-F1")
plt.plot(epochs, kappa, label="VAL kappa")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation metrics per epoch")
plt.legend()
plt.tight_layout()
plt.savefig("figures/val_metrics.png", dpi=200)

print("Saved: figures/train_loss.png, figures/val_metrics.png")

