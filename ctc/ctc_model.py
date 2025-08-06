import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCTransformer(nn.Module):
    def __init__(self, input_dim=354, model_dim=256, num_heads=4, num_layers=4, num_classes=50, dropout=0.2):
        super(CTCTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, num_classes + 1)  # +1 for CTC blank
        self.final_classifier = nn.Linear(model_dim, num_classes + 1)  # auxiliary head

        # Optional: encourage blank initially
        # with torch.no_grad():
        #     self.classifier.bias[-1].fill_(3.0)

    def forward(self, x):
        x = x.to(self.input_proj.weight.device)
        x_proj = self.input_proj(x)  # [B, T, D]

        # Mask padded frames (all-zero)
        mask = (x.abs().sum(dim=-1) == 0)  # [B, T]
        x_encoded = self.transformer(x_proj, src_key_padding_mask=mask)  # [B, T, D]

        # Main CTC logits
        logits = self.classifier(x_encoded)  # [B, T, V]
        print("Classifier logits stats: min", logits.min().item(), "max", logits.max().item())

        log_probs = F.log_softmax(logits, dim=-1)

        # Auxiliary prediction on last timestep
        aux_logits = self.final_classifier(x_encoded[:, -1])  # [B, V]
        return log_probs, aux_logits
