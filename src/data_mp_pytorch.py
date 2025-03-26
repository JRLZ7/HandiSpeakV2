import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np

class HandiSpeakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # Load the list of JSON files and create a label map
        self.json_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        self.label_map = {os.path.splitext(f)[0]: idx for idx, f in enumerate(self.json_files)}

        print("Label map:")
        print(self.label_map)

        print("Loaded JSON files:")
        print(self.json_files)

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = os.path.join(self.data_dir, self.json_files[idx])
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract word label from file name
        word = self.json_files[idx].replace('.json', '')
        label = self.label_map[word]

        # Combine keypoints for all videos of the word
        all_keypoints = []
        for video_id, video_data in data.items():
            video_keypoints = []
            for frame in video_data:
                frame_keypoints = []

                # Handle missing keypoints by adding zeroed points (keeping the same size)
                for part in ['face', 'pose', 'left_hand', 'right_hand']:
                    if part in frame:
                        keypoints = frame[part]
                        frame_keypoints.extend([kp['x'] for kp in keypoints])
                        frame_keypoints.extend([kp['y'] for kp in keypoints])
                        frame_keypoints.extend([kp['z'] for kp in keypoints])
                    else:
                        # Calculate how many zeroes to add based on expected number of keypoints
                        if part == 'face':
                            num_keypoints = 71  # Number of face keypoints we kept
                        elif part == 'pose':
                            num_keypoints = 4   # Number of pose keypoints we kept
                        else:  # Hands (left or right)
                            num_keypoints = 21  # Number of hand keypoints

                        frame_keypoints.extend([0.0] * 3 * num_keypoints)

                video_keypoints.append(frame_keypoints)

            # Convert list of frame keypoints to a tensor
            all_keypoints.append(torch.tensor(video_keypoints, dtype=torch.float32))

        # Stack all videos for the word (if multiple videos exist)
        keypoints_tensor = torch.cat(all_keypoints, dim=0)

        return keypoints_tensor, label


def collate_fn(batch):
    inputs, labels = zip(*batch)

    # Find the maximum length of the videos in the batch
    max_length = max(input_tensor.shape[0] for input_tensor in inputs)

    # Pad each video to the maximum length
    padded_inputs = []
    for input_tensor in inputs:
        padding_size = max_length - input_tensor.shape[0]
        padding = torch.zeros((padding_size, input_tensor.shape[1]))
        padded_tensor = torch.cat([input_tensor, padding], dim=0)
        padded_inputs.append(padded_tensor)

    # Stack padded videos into a single tensor
    padded_inputs = torch.stack(padded_inputs)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_inputs, labels


def create_dataloader(data_dir, batch_size=8, shuffle=True):
    dataset = HandiSpeakDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

# Test the DataLoader
data_dir = 'keypoints'  # Update to your keypoints directory
batch_size = 4
dataloader = create_dataloader(data_dir, batch_size)

for batch in dataloader:
    inputs, labels = batch
    print(f"Batch size: {inputs.shape}, Labels: {labels.shape}")
    break
