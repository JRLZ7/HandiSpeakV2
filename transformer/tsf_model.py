import torch
import torch.nn as nn

class HandiSpeakTransformer(nn.Module):
    def __init__(self, input_dim=354, model_dim=256, num_heads=8, num_layers=4, num_classes=50, dropout=0.1):
        super(HandiSpeakTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)  # [B, T, model_dim]
        x = self.transformer(x)  # [B, T, model_dim]
        x = x.mean(dim=1)  # Global average pooling over time
        out = self.classifier(x)
        return out
