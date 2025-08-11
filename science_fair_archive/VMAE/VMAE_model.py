import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification

class HandiSpeakModel(nn.Module):
    def __init__(self, num_classes):
        super(HandiSpeakModel, self).__init__()
        # Load pre-trained VideoMAE model for video classification
        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        # Adjust the classification head to match the number of classes
        self.model.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, x):
        # Ensure input has 16 frames (Time dimension)
        if x.shape[1] != 16:
            raise ValueError(f"Expected 16 frames, got {x.shape[1]} frames")
        
        # Forward pass through the VideoMAE model
        outputs = self.model(pixel_values=x)
        return outputs.logits

# Example usage
if __name__ == "__main__":
    num_classes = 50  # Adjust according to your dataset
    model = HandiSpeakModel(num_classes)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with dummy input of 16 frames
    dummy_input = torch.randn(1, 16, 3, 224, 224).to("cuda" if torch.cuda.is_available() else "cpu")
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected: (1, num_classes)
