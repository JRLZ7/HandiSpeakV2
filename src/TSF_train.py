import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from model import load_model
from data_processing import load_and_preprocess_data

def train_model(train_loader, val_loader, epochs=10):
    model = load_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            frames, labels = batch['frames'], batch['label']
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                frames, labels = batch['frames'], batch['label']
                outputs = model(frames)
                val_loss += criterion(outputs.logits, labels).item()
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = correct / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {val_loss}, Accuracy: {accuracy}")

    # Save the trained model
    torch.save(model.state_dict(), "timesformer_asl.pth")

if __name__ == "__main__":
    dataset = load_and_preprocess_data()
    train_loader = DataLoader(dataset['train'], batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset['validation'], batch_size=8)
    train_model(train_loader, val_loader)