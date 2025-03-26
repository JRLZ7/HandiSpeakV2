import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_mp_pytorch_test import create_dataloader
from LSTM_model import ASLClassifier

# Hyperparameters
input_size = 354  # Corrected input size
hidden_size = 128
num_layers = 2
num_classes = 20
batch_size = 8
num_epochs = 20
learning_rate = 0.001

# Initialize model, loss function, and optimizer
model = ASLClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare data loaders
data_dir = '/home/jason/Projects/HandiSpeakV2/keypoints_aug'
train_loader = create_dataloader(f"{data_dir}/train", batch_size)
val_loader = create_dataloader(f"{data_dir}/val", batch_size)

best_val_acc = 0.0  # Track the best validation accuracy

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Training Phase
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    train_acc = 100 * correct / total
    train_loss = total_loss / len(train_loader)

    # Validation Phase
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            val_loss += loss.item()

    val_acc = 100 * val_correct / val_total
    val_loss /= len(val_loader)

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_asl_lstm_model.pth')
        print(f"✅ New best model saved with validation accuracy: {best_val_acc:.2f}%")

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

print("Training complete!")
