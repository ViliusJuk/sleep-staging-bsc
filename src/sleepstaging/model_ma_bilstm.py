import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation (kanalų dėmesys) mažam CNNui."""
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Linear(c, max(1, c // r))
        self.fc2 = nn.Linear(max(1, c // r), c)

    def forward(self, x):  # x: (B,C,T)
        s = x.mean(-1)                    # (B,C)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))    # (B,C)
        s = s.unsqueeze(-1)               # (B,C,1)
        return x * s

class MultiScaleCNN(nn.Module):
    """
    Multi-Scale CNN: keli lygiagretūs Conv1d (kernels: 3,5,7) + SE attention.
    Grąžina fiksuoto ilgio epochos vektorių (feat_dim).
    """
    def __init__(self, in_ch=1, base=32, feat_dim=128):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(in_ch, base, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.b2 = nn.Sequential(nn.Conv1d(in_ch, base, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        self.b3 = nn.Sequential(nn.Conv1d(in_ch, base, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2))
        self.merge = nn.Sequential(
            nn.Conv1d(base*3, base*3, 1), nn.ReLU(),
            nn.Conv1d(base*3, base*2, 3, padding=1), nn.ReLU(),
        )
        self.se = SEBlock(base*2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base*2, feat_dim)

    def forward(self, x):   # (B,1,T)
        h = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)  # (B, base*3, T/2)
        h = self.merge(h)                                          # (B, base*2, T/2)
        h = self.se(h)                                             # attention
        h = self.pool(h).squeeze(-1)                               # (B, base*2)
        return self.fc(h)                                          # (B, feat_dim)

class MA_CNN_BiLSTM(nn.Module):
    """
    MA-CNN-BiLSTM: Multi-Scale Attention CNN (per epochą) -> BiLSTM (per langą).
    Įvestis: (B, W, 1, T); išvestis: (B, n_classes).
    """
    def __init__(self, n_classes=5, feat_dim=128, lstm_hidden=128, lstm_layers=1, bidirectional=True, dropout=0.1):
        super().__init__()
        self.cnn = MultiScaleCNN(in_ch=1, base=32, feat_dim=feat_dim)
        self.lstm = nn.LSTM(
            input_size=feat_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=bidirectional, dropout=dropout if lstm_layers > 1 else 0.0
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):  # x: (B,W,1,T)
        B, W, C, T = x.shape
        x = x.view(B*W, C, T)
        feats = self.cnn(x)              # (B*W, feat_dim)
        feats = feats.view(B, W, -1)     # (B, W, feat_dim)
        out, _ = self.lstm(feats)        # (B, W, H*)
        center = out[:, out.size(1)//2, :]
        return self.head(center)         # (B, n_classes)
