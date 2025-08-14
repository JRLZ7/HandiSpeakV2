import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from LSTM.LSTM_model import ASLClassifier
from LSTM.LSTM_data import create_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt

# NEW: metrics imports
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 2.8e-3 # change for results
# 1, 2, 2.45, 2.87
NUM_CLASSES = 200

# TWO HUNDRED WORDS = 86.20% // 38.25%
# ONE FIFTY WORDS = 92.53% // 70.87%
# ONE HUNDRED WORDS = 93.6% (91.4) // 89.80%
# FIFTY WORDS = 94.00%

# Load dataset
train_loader, word_to_index = create_dataloader(data_dir="keypoints_aug_200/train", batch_size=BATCH_SIZE)
val_loader, _ = create_dataloader(data_dir="keypoints_aug_200/val", batch_size=BATCH_SIZE, shuffle=False, word_to_index=word_to_index)

# Initialize model, loss, and optimizer
model = ASLClassifier(input_size=354, hidden_size=256, num_layers=2, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Tracking variables
train_accs = []
val_accs = []
train_losses = []
val_losses = []
best_val_acc = 0.0

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")
    for batch in loop:
        keypoints, labels = batch
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

        loop.set_postfix(loss=loss.item(), accuracy=100. * correct / total)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for keypoints, labels in val_loader:
            keypoints, labels = keypoints.to(device), labels.to(device)
            outputs = model(keypoints)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100. * val_correct / val_total
    train_accs.append(100. * correct / total)
    val_accs.append(val_accuracy)
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    # Save best model (defer writing until after training)
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_model_state = model.state_dict()
        best_epoch = epoch + 1

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Acc: {100. * correct / total:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Val Acc: {val_accuracy:.2f}%")

# === Save best model ===
os.makedirs("200_results", exist_ok=True)
os.makedirs("models", exist_ok=True)  # NEW: ensure folder exists
torch.save(best_model_state, f"models/best_lstm_epoch{best_epoch}_acc{best_val_acc:.2f}.pt")
print(f"✅ Saved best model from epoch {best_epoch} with val acc {best_val_acc:.2f}%")

# Plot accuracy
plt.figure()
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy over Epochs (200 words)')
plt.savefig('200_results/LSTM_accuracy_plot.png')

# Plot loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs (200 words)')
plt.savefig('200_results/LSTM_loss_plot.png')

# === Confusion Matrix + Metrics ===
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

cm = confusion_matrix(all_labels, all_preds)
np.save("200_results/LSTM_confusion_matrix.npy", cm)  # optional but handy for later

fig, ax = plt.subplots(figsize=(15, 14))  # 🔼 More space for 100 words
display_labels = list(range(1, len(word_to_index) + 1))  # Use class numbers 1–100
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(ax=ax, xticks_rotation=90, include_values=False, colorbar=True, cmap="viridis")

# 🔼 Increase font sizes
plt.xticks(fontsize=90)
plt.yticks(fontsize=90)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.set_xlabel("Predicted Label", fontsize=23)
ax.set_ylabel("True Label", fontsize=23)
plt.title("Confusion Matrix (200 Words)", fontsize=50)
plt.tight_layout()
plt.savefig("200_results/LSTM_confusion_matrix.png", dpi=300)

# NEW: Precision/Recall/F1 (macro & weighted) — computed from labels/preds
report_str = classification_report(all_labels, all_preds, digits=3)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

print("\n=== Classification Report (per-class + averages) ===")
print(report_str)
print(f"Macro Precision: {prec_macro:.3f} | Macro Recall: {rec_macro:.3f} | Macro F1: {f1_macro:.3f}")
print(f"Weighted Precision: {prec_weight:.3f} | Weighted Recall: {rec_weight:.3f} | Weighted F1: {f1_weight:.3f}")

with open("200_results/LSTM_classification_report.txt", "w") as f:
    f.write(report_str + "\n")
    f.write(f"\nMacro P/R/F1: {prec_macro:.3f}/{rec_macro:.3f}/{f1_macro:.3f}\n")
    f.write(f"Weighted P/R/F1: {prec_weight:.3f}/{rec_weight:.3f}/{f1_weight:.3f}\n")