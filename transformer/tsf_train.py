import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tsf_data import KeypointDataset
from tsf_model import HandiSpeakTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 8
learning_rate = 1.78e-4
num_epochs = 20
num_classes = 50

# TWO HUNDRED WORDS = 88.15%
# ONE HUNDRED FIFTY WORDS = 91.00%
# ONE HUNDRED WORDS = 92.10%
# FIFTY WORDS = 93.60%


# ✅ Custom collate function to stack
def collate_fn(batch):
    keypoints = torch.stack([item[0] for item in batch])  # [B, T, F]
    labels = torch.tensor([item[1] for item in batch])
    return keypoints, labels

# Dataset and DataLoaders
train_dataset = KeypointDataset("keypoints_aug_50/train")
val_dataset = KeypointDataset("keypoints_aug_50/val")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Model, loss, optimizer
model = HandiSpeakTransformer(input_dim=354, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_accs = []
val_accs = []
train_losses = []
val_losses = []
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for keypoints, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        keypoints, labels = keypoints.to(device), labels.to(device)
        outputs = model(keypoints)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for keypoints, labels in val_loader:
            keypoints, labels = keypoints.to(device), labels.to(device)
            outputs = model(keypoints)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100. * val_correct / val_total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_state = model.state_dict()
        best_epoch = epoch + 1

# Save best model
os.makedirs("50_results", exist_ok=True)
torch.save(best_model_state, f"models/Transformer_best_epoch{best_epoch}_acc{best_acc:.2f}.pt")
print(f"✅ Saved best model from epoch {best_epoch} with val acc {best_acc:.2f}%")

# Plot accuracy
plt.figure()
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy over Epochs (50 words)')
plt.savefig('50_results/TSF_accuracy_plot.png')

# Plot loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs (50 words)')
plt.savefig('50_results/TSF_loss_plot.png')

# Confusion Matrix
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for keypoints, labels in val_loader:
        keypoints, labels = keypoints.to(device), labels.to(device)
        outputs = model(keypoints)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# np.save("models/tsf_preds.npy", np.array(all_preds))
# np.save("models/tsf_labels.npy", np.array(all_labels))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(15, 14))  # Match LSTM

# Replace labels with class numbers 1–num_classes
display_labels = list(range(1, num_classes + 1))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(
    ax=ax,
    xticks_rotation=90,
    include_values=False,
    colorbar=True,
    cmap="viridis"
)

plt.xticks(fontsize=90)
plt.yticks(fontsize=90)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_xlabel("Predicted Label", fontsize=23)
ax.set_ylabel("True Label", fontsize=23)
plt.title("Confusion Matrix (50 Words)", fontsize=50)
plt.tight_layout()
plt.savefig("50_results/TSF_confusion_matrix.png", dpi=300)