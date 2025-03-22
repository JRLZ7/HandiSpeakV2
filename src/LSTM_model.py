import torch
import torch.nn as nn

class HandiSpeakLSTM(nn.Module):
    def __init__(self, input_size=258, hidden_size=128, num_layers=2, num_classes=20):
        super(HandiSpeakLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the output from the last time step
        out = self.fc(out)
        return out

# Example usage
if __name__ == "__main__":
    model = HandiSpeakLSTM()
    dummy_input = torch.randn(1, 60, 258)  # Batch size 1, 60 frames, 258 features
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected: (1, 20)
