import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from data_pytorch import HandiSpeakDataset
from VMAE_model import HandiSpeakModel

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
num_classes = 50
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

# Datasets and loaders
train_dataset = HandiSpeakDataset("~/Projects/HandiSpeakV2/datasets/top_50_metadata/train", transform=transform)
val_dataset = HandiSpeakDataset("~/Projects/HandiSpeakV2/datasets/top_50_metadata/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Tracking
train_accs = []
val_accs = []
train_losses = []
val_losses = []

best_acc = 0.0

# Training
def train_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        frames, labels = frames.to(device), labels.to(device)
        outputs = model(frames)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    return avg_loss, acc

def validate_epoch(epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    avg_loss = running_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")
    return avg_loss, acc

# Training loop
model_dir = os.path.expanduser("~/Projects/HandiSpeakV2/models")
os.makedirs(model_dir, exist_ok=True)
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(epoch)
    val_loss, val_acc = validate_epoch(epoch)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = model.state_dict()
        best_epoch = epoch + 1

# Save best model
torch.save(best_model_state, os.path.join(model_dir, f"VMAE_best_epoch{best_epoch}_acc{best_acc:.2f}.pth"))
print(f"✅ Best model saved from epoch {best_epoch} with val acc {best_acc:.2f}%")

# Accuracy plot
plt.figure()
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Validation Accuracy - VideoMAE')
plt.savefig(os.path.join(model_dir, "VMAE_accuracy_plot.png"))

# Loss plot
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss - VideoMAE')
plt.savefig(os.path.join(model_dir, "VMAE_loss_plot.png"))

# Confusion matrix on val set
all_preds, all_labels = [], []
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    for frames, labels in val_loader:
        frames, labels = frames.to(device), labels.to(device)
        outputs = model(frames)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, xticks_rotation=90)
plt.title("Confusion Matrix - VideoMAE")
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "VMAE_confusion_matrix.png"))

print("✅ Training complete and all plots saved!")
