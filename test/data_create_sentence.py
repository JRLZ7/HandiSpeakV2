import os
import json
import random
from pathlib import Path

from pathlib import Path

word_json_dir = Path("/home/jason/Projects/HandiSpeakV2/keypoints_aug_50")
output_dir = Path("synthetic_sentences_50")
output_dir.mkdir(exist_ok=True)

num_sentences = 500
min_words, max_words = 5, 8
padding_frames = 3  # zero-pause between signs

# Load word data
word_to_index = {}
word_keypoints = {}

for json_path in word_json_dir.glob("*.json"):
    word = json_path.stem
    with open(json_path) as f:
        video_data = json.load(f)
    if video_data:
        word_to_index[word] = len(word_to_index)
        word_keypoints[word] = list(video_data.values())

def zero_frame():
    return [0.0] * 354

synthetic_data = []

for i in range(num_sentences):
    sentence_length = random.randint(min_words, max_words)
    words = random.choices(list(word_keypoints.keys()), k=sentence_length)

    frames = []
    label = []

    for word in words:
        label.append(word_to_index[word])
        sequence = random.choice(word_keypoints[word])
        for frame in sequence:
            flattened = []
            for group in ['face', 'pose', 'left_hand', 'right_hand']:
                for kp in frame[group]:
                    flattened.extend([kp['x'], kp['y'], kp['z']])
            flattened = [val + random.gauss(0, 0.001) for val in flattened]
            frames.append(flattened)

        frames.extend([zero_frame()] * padding_frames)

    synthetic_data.append({
        "frames": frames,
        "label": label
    })

# Save JSON file
with open(output_dir / "synthetic_50_sentences.json", "w") as f:
    json.dump(synthetic_data, f, indent=2)

print("✅ Saved 500 synthetic sentences.")
