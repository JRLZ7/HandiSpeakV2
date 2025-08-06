import json
import torch
from torch.utils.data import Dataset

class CTCDataset(Dataset):
    def __init__(self, json_path, max_frames=100):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        frames = sample["frames"]
        label = sample["label"]

        if len(frames) < self.max_frames:
            pad = [[0.0] * 354] * (self.max_frames - len(frames))
            frames += pad
        else:
            frames = frames[:self.max_frames]

        frames_tensor = torch.tensor(frames, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return frames_tensor, label_tensor
