import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from LSTM_model import ASLClassifier
from data_mp_pytorch import create_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_CLASSES = 20

# Load dataset
train_loader, word_to_index = create_dataloader(data_dir="keypoints_aug/train", batch_size=BATCH_SIZE)
val_loader, _ = create_dataloader(data_dir="keypoints_aug/val", batch_size=BATCH_SIZE, shuffle=False, word_to_index=word_to_index)

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

    # Save best model
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/best_lstm_epoch{epoch+1}_acc{val_accuracy:.2f}.pt")

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f}, "
          f"Train Acc: {100. * correct / total:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}, "
          f"Val Acc: {val_accuracy:.2f}%")

# Plot accuracy
plt.figure()
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy over Epochs')
plt.savefig('models/accuracy_plot.png')

# Plot loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.savefig('models/loss_plot.png')

# Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

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

np.save("models/LSTM_preds.npy", np.array(all_preds))
np.save("models/LSTM_labels.npy", np.array(all_labels))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=word_to_index.keys())
disp.plot(xticks_rotation=90)
plt.title("Confusion Matrix")
plt.savefig("models/confusion_matrix.png")
