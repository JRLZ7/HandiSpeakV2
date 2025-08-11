import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Sinusoidal PE → no learned params, ONNX/quantization friendly, handles variable lengths."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                       # [L, D]
        position = torch.arange(0, max_len).unsqueeze(1).float() # [L, 1]
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))              # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        if T > self.pe.size(1):
            # Safety extend (rare) so very long synthetic sentences won’t crash.
            self._extend(T, x.device, x.size(-1))
        return x + self.pe[:, :T, :]

    def _extend(self, new_L: int, device, d_model: int):
        pos = torch.arange(0, new_L, device=device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(new_L, d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

class TSFCTCEncoder(nn.Module):
    """
    Transformer encoder that outputs per-timestep logits for CTC:
      - Input LayerNorm + Linear → robust to feature scale, easy model swaps later.
      - Sinusoidal PE → stable, portable.
      - Pre-norm encoder layers (PyTorch default) → smoother gradients.
      - No pooling → CTC needs one logit per timestep.
      - Post LayerNorm → steadier logits across variable T.
    """
    def __init__(self,
                 feature_dim: int = 354,   # F (MediaPipe), fixed across your pipeline
                 num_classes: int = 51,    # 50 vocab + 1 blank (id 0)
                 d_model: int = 256,
                 nhead: int = 8,           # 256/8=32 per head → typical sweet spot
                 num_layers: int = 4,      # start moderate; scale if needed
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.in_norm = nn.LayerNorm(feature_dim)
        self.in_proj = nn.Linear(feature_dim, d_model)
        self.pos = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.post_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    @staticmethod
    def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        # True where padding, False where valid. Shape [B, T].
        rng = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return rng >= lengths.unsqueeze(1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 354], lengths: [B]
        h = self.in_norm(x.float())          # stabilize per-feature scale
        h = self.in_proj(h)                  # [B, T, D]
        h = self.pos(h)                      # add positional info
        pad_mask = self.make_pad_mask(lengths, h.size(1))
        h = self.encoder(h, src_key_padding_mask=pad_mask)  # ignore pads in attention
        h = self.post_norm(h)
        logits = self.classifier(h)          # [B, T, 51]
        return logits
