import argparse
import torch

from src.sleepstaging.model_bilstm import CNNBiLSTM
from src.sleepstaging.labels import CLASSES

def strip_module_prefix(sd):
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="state_dict .pth")
    ap.add_argument("--out", required=True, help="TorchScript output .pt")
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--T", type=int, default=3000)  # 30s*100Hz
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    sd = strip_module_prefix(sd)

    model = CNNBiLSTM(
        n_classes=len(CLASSES),
        lstm_hidden=args.hidden,
        lstm_layers=args.layers,
        dropout=args.dropout,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Trace sequence mode: (B, L, T)
    example = torch.randn(1, args.seq_len, args.T)
    ts = torch.jit.trace(model, example)
    ts.save(args.out)

    with torch.no_grad():
        y = ts(example)
    print("OK. output shape:", tuple(y.shape))
    print("Saved:", args.out)

if __name__ == "__main__":
    main()

