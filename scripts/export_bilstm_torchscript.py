import argparse
import torch

# 1) Importuok savo BiLSTM klasę
# Pakeisk importą pagal tavo projektą (labai tikėtina, kad taip):
from src.sleepstaging.model_bilstm import BiLSTM  # <-- jei klasė vadinasi kitaip, pakeisk

def strip_module_prefix(state_dict):
    # jei treniruota su DataParallel (raktai prasideda "module.")
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Kelias iki checkpoint (.pt/.pth)")
    ap.add_argument("--out", required=True, help="Kur išsaugoti TorchScript modelį (.pt)")
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--T", type=int, default=3000)  # 30s * 100Hz = 3000
    ap.add_argument("--n-classes", type=int, default=5)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # 2) Ištraukiam state_dict iš checkpoint (skirtingi projektai saugo skirtingai)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and any(k.endswith("weight") for k in ckpt.keys()):
        sd = ckpt
    else:
        raise RuntimeError(f"Neatpažinau checkpoint formato. Raktai: {list(ckpt.keys())[:20]}")

    sd = strip_module_prefix(sd)

    # 3) Sukuriam modelį
    # !!! Čia gali tekti suderinti init argumentus pagal tavo BiLSTM klasę !!!
    # Minimalus variantas: BiLSTM(n_classes=5, ...)
    model = BiLSTM(n_classes=args.n_classes)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # 4) TorchScript: trace su example input
    # Čia darom prielaidą, kad tavo modelis priima [B, L, T]
    example = torch.randn(1, args.seq_len, args.T)
    ts = torch.jit.trace(model, example)
    ts.save(args.out)

    # 5) Greitas sanity check
    with torch.no_grad():
        y = ts(example)
    print("OK. Output shape:", tuple(y.shape))
    print("Saved to:", args.out)

if __name__ == "__main__":
    main()

