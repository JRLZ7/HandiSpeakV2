import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ctc_data import CTCDataset
from ctc_model import CTCTransformer
from wer import compute_wer
from tqdm import tqdm
import numpy as np

# Paths and parameters
train_path = "synthetic_sentences_50/synthetic_50_train.json"
val_path = "synthetic_sentences_50/synthetic_50_val.json"

epochs = 20
batch_size = 1
learning_rate = 1e-4
num_classes = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and Dataloader
def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lengths = torch.tensor([len(seq) for seq in inputs])
    target_lengths = torch.tensor([len(seq) for seq in targets])
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = torch.cat(targets)
    return inputs, targets, input_lengths, target_lengths

full_dataset = CTCDataset(train_path)
train_dataset = torch.utils.data.Subset(full_dataset, [0])  # only the first sample

frames, label = train_dataset[0]
print("Frames shape:", frames.shape)
print("Label:", label.tolist())
print("First frame slice:", frames[0][:10])
print("Last frame slice:", frames[-1][:10])

val_dataset = CTCDataset(val_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Model, optimizer, loss
model = CTCTransformer(model_dim=256, num_heads=4, num_layers=4).to(device)
print("Classifier weight mean:", model.classifier.weight.mean().item())
print("Classifier bias mean:", model.classifier.bias.mean().item())

criterion = nn.CTCLoss(blank=num_classes, reduction='mean', zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load pre-trained weights (excluding classifier)
pretrained_path = "/home/jason/Projects/HandiSpeakV2/models/95.8_best_50_TSF.pt"
pretrained_state = torch.load(pretrained_path, map_location=device)
filtered_state = {k: v for k, v in pretrained_state.items() if not k.startswith("classifier")}
missing, unexpected = model.load_state_dict(filtered_state, strict=False)
print("✅ Pretrained loaded (excluding classifier).")
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for inputs, targets, input_lengths, target_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs = inputs.to(device)
        # Debug: count non-zero (non-padded) frames
        nonzero_frames = (inputs.abs().sum(dim=-1) > 0).sum().item()
        total_frames = inputs.size(0) * inputs.size(1)
        print(f"🧪 Nonzero frames: {nonzero_frames} / {total_frames}")

        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        log_probs, aux_logits = model(inputs)  # [B, T, V], [B, V]
        log_probs = log_probs.transpose(0, 1)  # [T, B, V]

        # CTC loss
        ctc_loss = criterion(log_probs, targets, input_lengths, target_lengths)

        # Auxiliary loss (last label of each target sequence)
        last_labels = []
        start = 0
        for length in target_lengths:
            end = start + length.item()
            last_labels.append(targets[end - 1].item())
            start = end

        last_labels = torch.tensor(last_labels, device=device)

        aux_loss = nn.functional.cross_entropy(aux_logits, last_labels, label_smoothing=0.1)

        total_loss = ctc_loss + 0.1 * aux_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()

    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    # Evaluate WER on validation set
    model.eval()
    total_wer = 0.0
    count = 0
    print(f"\n📋 Sample Predictions (Epoch {epoch+1}):")

    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            log_probs, _ = model(inputs)
            pred = torch.argmax(log_probs, dim=-1)

            for i in range(pred.shape[0]):
                pred_seq = torch.unique_consecutive(pred[i])
                pred_seq = [p.item() for p in pred_seq if p.item() != num_classes]

                tgt_len = target_lengths[i].item()
                tgt_start = sum(target_lengths[:i])
                tgt_end = tgt_start + tgt_len
                ref_seq = targets[tgt_start:tgt_end].tolist()

                if i < 3:
                    print("REF :", ref_seq)
                    print("PRED:", pred_seq)
                
                log_probs, _ = model(inputs)
                pred = torch.argmax(log_probs, dim=-1)
                print("Raw predicted indices per timestep:", pred.tolist())


                total_wer += compute_wer(ref_seq, pred_seq)

    avg_wer = total_wer / len(val_dataset)
    print(f"✅ Val WER after Epoch {epoch+1}: {avg_wer:.4f}")
