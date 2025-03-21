import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification
import torchvision.transforms as transforms
import os

from data_pytorch import HandiSpeakDataset
from model import HandiSpeakModel

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
num_classes = 20
num_epochs = 10
learning_rate = 1e-4
batch_size = 4

# Model initialization
model = HandiSpeakModel(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Datasets and DataLoaders
train_dataset = HandiSpeakDataset("~/Projects/HandiSpeakV2/datasets/20_words/train", transform=transform)
val_dataset = HandiSpeakDataset("~/Projects/HandiSpeakV2/datasets/20_words/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Training function
def train_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        frames, labels = frames.to(device), labels.to(device)

        # Forward pass
        outputs = model(frames)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Validation function
def validate_epoch(epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
            frames, labels = frames.to(device), labels.to(device)

            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)

            # Calculate metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    print(f"Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    return epoch_acc

# Training loop
best_acc = 0.0
for epoch in range(num_epochs):
    train_epoch(epoch)
    val_acc = validate_epoch(epoch)

    # Save the best model
    if val_acc > best_acc:
        best_acc = val_acc
        model_dir = os.path.expanduser("~/Projects/HandiSpeakV2/models")
        os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
        torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
        print(f"✅ New best model saved with accuracy: {best_acc:.2f}%")

print("Training complete!")
