import torch
import torch.nn as nn


class SleepTransformer(nn.Module):
    """
    Epoch-based Transformer model:
    įvestis: (B, 1, 3000) Fpz-Cz epochos
    1) Conv1d -> "patch embedding"
    2) Positional encoding
    3) TransformerEncoder
    4) vidurkinam per laiką -> klasifikacija
    """

    def __init__(
        self,
        n_classes: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Patch embedding: iš 3000 taškų -> ~300 tokenų (seq_len ~ T')
        # stride=10 -- gana saugu, kad atmintis nesprogtų
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=25,
            stride=10,
            padding=12,
        )  # (B, d_model, T')

        # Simple learnable positional encoding (T' pozicijoms)
        self.pos_dropout = nn.Dropout(dropout)
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, C)
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Galva: vidurkis per laiką + klasifikacija
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        # Positional embedding su fixed max seq_len
        self.max_len = 400  # užtenka 3000/10=300 + rezervas
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 3000)
        # 1) Patch embedding
        z = self.patch_embed(x)          # (B, d_model, T')
        z = z.transpose(1, 2)            # (B, T', d_model)
        B, T, C = z.shape

        if T > self.max_len:
            # saugumo sumetimais cropinam, bet 3000/10 ~ 300, tai neturėtų prireikt
            z = z[:, : self.max_len, :]
            T = self.max_len

        # 2) Positional encoding
        pos = self.pos_embed[:, :T, :]
        z = z + pos
        z = self.pos_dropout(z)

        # 3) Transformer encoder
        h = self.encoder(z)              # (B, T, d_model)

        # 4) Global average pooling per laiką
        h_mean = h.mean(dim=1)           # (B, d_model)

        # 5) Klasifikacija (raw logits)
        logits = self.head(h_mean)       # (B, n_classes)
        return logits

