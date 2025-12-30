import torch
import torch.nn as nn


class SeqTransformer(nn.Module):
    """
    Sequence-of-epochs Transformer.

    Input:
      x: (B, L, 3000)   where L = seq_len (epochs), each epoch is 3000 samples (1 channel).
    Output:
      logits: (B, n_classes) for CENTER epoch label (L//2), same as SleepEDFSequenceDataset(y_center).

    Pipeline:
      - Per-epoch CNN embedding -> (B, L, d_model)
      - Positional embedding over L
      - TransformerEncoder over L
      - Take center token -> classifier head
    """

    def __init__(
        self,
        n_classes: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()

        # Per-epoch feature extractor: (B*L, 1, 3000) -> (B*L, d_model)
        self.epoch_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),  # 3000 -> ~1500
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # ~750

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # ~375
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2),  # ~187

            nn.Conv1d(64, d_model, kernel_size=3, stride=2, padding=1),  # ~94
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )

        self.max_seq_len = max_seq_len
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, L, C)
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 3000)
        if x.ndim != 3:
            raise ValueError(f"Expected x (B,L,3000), got shape={tuple(x.shape)}")

        B, L, T = x.shape
        if L > self.max_seq_len:
            # crop if someone sets seq_len > max_seq_len
            x = x[:, : self.max_seq_len, :]
            L = self.max_seq_len

        # CNN embed each epoch
        x = x.reshape(B * L, 1, T)                 # (B*L, 1, 3000)
        z = self.epoch_cnn(x)                      # (B*L, d_model, T')
        z = z.mean(dim=-1)                         # (B*L, d_model)
        z = z.reshape(B, L, -1)                    # (B, L, d_model)

        # Positional embedding
        z = z + self.pos_embed[:, :L, :]
        z = self.pos_dropout(z)

        # Transformer over epoch sequence
        h = self.encoder(z)                        # (B, L, d_model)

        # Center token classification
        center = L // 2
        hc = h[:, center, :]                       # (B, d_model)
        logits = self.head(hc)                     # (B, n_classes)
        return logits

