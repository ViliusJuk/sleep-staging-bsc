import argparse
import torch

from src.sleepstaging.model_bilstm import BiLSTM  # jei klasė kitaip vadinasi – pakeisk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=20)
    ap.add_argument("--T", type=int, default=3000)
    args = ap.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    # jei kartais būtų DataParallel:
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # SVARBU: init turi sutapti su train_bilstm.py
    model = BiLSTM(n_classes=5)
    model.load_state_dict(sd, strict=True)
    model.eval()

    example = torch.randn(1, args.seq_len, args.T)  # [B,L,T]
    ts = torch.jit.trace(model, example)
    ts.save(args.out)

    with torch.no_grad():
        y = ts(example)
    print("OK output:", tuple(y.shape))
    print("Saved:", args.out)

if __name__ == "__main__":
    main()

