import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import numpy as np  # 🔥 Fix: Import NumPy


class HandiSpeakDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.data = []

        # Collect all video folders and frames
        for label in self.classes:
            label_dir = os.path.join(self.root_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for video_folder in sorted(os.listdir(label_dir)):
                video_path = os.path.join(label_dir, video_folder)
                if os.path.isdir(video_path):
                    # Use glob to find all frames (sorted order)
                    frame_files = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
                    if len(frame_files) > 0:
                        self.data.append((frame_files, self.classes.index(label)))

        print(f"Loaded {len(self.data)} video samples from {self.root_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_files, label = self.data[idx]

        # Sample 16 frames from 60 frames (evenly spaced)
        total_frames = len(frame_files)
        sample_indices = np.linspace(0, total_frames - 1, 16).astype(int)
        sampled_frames = [frame_files[i] for i in sample_indices]

        # Load and process each frame
        frames = []
        for frame_file in sampled_frames:
            image = Image.open(frame_file).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        # Stack frames along the time dimension (T, C, H, W)
        frames = torch.stack(frames)

        return frames, label


# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = HandiSpeakDataset("~/Projects/HandiSpeakV2/datasets/20_words/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Test the dataloader
    for batch in dataloader:
        frames, labels = batch
        print(f"Batch frames shape: {frames.shape}")  # Should be (batch_size, T, C, H, W)
        print(f"Batch labels: {labels}")
        break
