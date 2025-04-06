import torch
import torch.nn as nn

class ASLClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ASLClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256 * 2, num_classes)  # correct


    def forward(self, x):
        out, _ = self.lstm(x)
        # print(f"[DEBUG] LSTM output shape: {out.shape}")  # [batch, seq_len, hidden*2]
        out = out[:, -1, :]  # Get the output from the last frame
        out = self.fc(out)   # [batch, num_classes]
        return out


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    input_size = 354  # Each frame has 354 keypoints (x, y, z flattened)
    hidden_size = 128
    num_layers = 2
    
    # PLEASE CHANGE IF UR DOING DIFFERENT DATASET !!!!!!!!!!!!!!!!!!
    num_classes = 50

    sequence_length = 20  # 20 frames per video
    batch_size = 1

    model = ASLClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

    # Dummy input
    dummy_input = torch.randn(batch_size, sequence_length, input_size).to(device)

    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected: (1, 20)