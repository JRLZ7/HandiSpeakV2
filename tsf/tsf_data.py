import os
import json
import torch
from torch.utils.data import Dataset
from glob import glob

class KeypointDataset(Dataset):
    def __init__(self, directory, max_frames=20, word_to_index=None):
        self.samples = []
        self.max_frames = max_frames
        self.word_to_index = word_to_index
        all_words = set()

        json_files = glob(os.path.join(directory, '*.json'))

        for word_path in json_files:
            word = os.path.splitext(os.path.basename(word_path))[0]
            all_words.add(word)

            with open(word_path, 'r') as f:
                video_dict = json.load(f)
                for video_id, frames in video_dict.items():
                    self.samples.append({
                        'word': word,
                        'video_id': video_id,
                        'frames': frames
                    })

        if self.word_to_index is None:
            self.word_to_index = {word: idx for idx, word in enumerate(sorted(all_words))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        word = sample['word']
        frames = sample['frames']

        # Pad or truncate to fixed length
        if len(frames) < self.max_frames:
            pad = [self._zero_frame()] * (self.max_frames - len(frames))
            frames += pad
        else:
            frames = frames[:self.max_frames]

        keypoints = []
        for frame in frames:
            flattened = []
            for group in ['face', 'pose', 'left_hand', 'right_hand']:
                for kp in frame[group]:
                    # ✅ Include all 3 coordinates
                    flattened.extend([kp['x'], kp['y'], kp['z']])
            keypoints.append(flattened)

        keypoints_tensor = torch.tensor(keypoints, dtype=torch.float32)
        label = self.word_to_index[word]
        return keypoints_tensor, label


    def _zero_frame(self):
        return {
            'face': [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(72)],
            'pose': [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(4)],
            'left_hand': [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(21)],
            'right_hand': [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(21)]
        }
        
