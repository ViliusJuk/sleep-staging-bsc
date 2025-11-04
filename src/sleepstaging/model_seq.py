# src/sleepstaging/model_seq.py
import torch
import torch.nn as nn

class EpochCNN(nn.Module):
    """
    Maža 1D CNN požymiams iš vienos epochos (1×T) išgauti.
    Gražina fiksuoto ilgio vektorių (feat_dim).
    """
    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B,128,1)
        )
        self.fc = nn.Linear(128, feat_dim)

    def forward(self, x):  # x: (B,1,T)
        h = self.net(x).squeeze(-1)   # (B,128)
        return self.fc(h)             # (B,feat_dim)

class CNNGRU(nn.Module):
    """
    TinySleepNet-style: epochų CNN -> GRU per langą -> klasifikacija.
    Įvestis: (B, W, 1, T)
    Išvestis: (B, n_classes)
    """
    def __init__(self, n_classes=5, feat_dim=128, hidden=128, num_layers=1, bidirectional=True, dropout=0.1):
        super().__init__()
        self.cnn = EpochCNN(feat_dim=feat_dim)
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):  # x: (B,W,1,T)
        B, W, C, T = x.shape
        x = x.view(B * W, C, T)
        feats = self.cnn(x)          # (B*W, feat)
        feats = feats.view(B, W, -1) # (B, W, feat)
        out, _ = self.gru(feats)     # (B, W, H*)
        center = out[:, out.size(1)//2, :]  # tik vidurinės epochos atstovavimas
        logits = self.head(center)   # (B, n_classes)
        return logits
