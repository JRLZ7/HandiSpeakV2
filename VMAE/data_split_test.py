import cv2
import os
import json
import numpy as np
import random
from tqdm import tqdm

# Parameters
desired_fps = 30
desired_frames = 60
frame_size = (224, 224)
train_ratio = 0.8

# Dataset name (controls metadata & output paths)
dataset_name = "top_50_metadata"

# Paths
metadata_path = os.path.expanduser(f"~/Projects/HandiSpeakV2/datasets/{dataset_name}.json")
video_root = os.path.expanduser("~/Projects/HandiSpeakV2/data/WLASL/videos")
train_dir = os.path.expanduser(f"~/Projects/HandiSpeakV2/datasets/{dataset_name}/train")
val_dir = os.path.expanduser(f"~/Projects/HandiSpeakV2/datasets/{dataset_name}/val")

# Create output directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Load metadata
with open(metadata_path, "r") as file:
    metadata = json.load(file)

# Normalize frame count
def normalize_frames(frames):
    frame_count = len(frames)
    if frame_count > desired_frames:
        indices = np.linspace(0, frame_count - 1, desired_frames).astype(int)
        return [frames[i] for i in indices]
    elif frame_count < desired_frames:
        return frames + [frames[-1]] * (desired_frames - frame_count)
    return frames

# Extract and save frames from a video
def extract_and_save_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(fps / desired_fps))

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
        count += 1
    cap.release()

    frames = normalize_frames(frames)
    for i, frame in enumerate(frames):
        frame_filename = f"frame_{i:04d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

# Main loop
for word, video_list in tqdm(metadata.items(), desc="Splitting and Extracting Data"):
    word_train_dir = os.path.join(train_dir, word)
    word_val_dir = os.path.join(val_dir, word)
    os.makedirs(word_train_dir, exist_ok=True)
    os.makedirs(word_val_dir, exist_ok=True)

    random.shuffle(video_list)
    split_point = int(len(video_list) * train_ratio)
    train_videos = video_list[:split_point]
    val_videos = video_list[split_point:]

    for video_name in train_videos:
        video_path = os.path.join(video_root, video_name)
        video_output_dir = os.path.join(word_train_dir, os.path.splitext(video_name)[0])

        if os.path.exists(os.path.join(video_output_dir, "frame_0000.jpg")):
            print(f"⏩ Skipping {video_name} (already extracted)")
            continue

        if not os.path.exists(video_path):
            print(f"❌ Missing video: {video_path}")
            continue

        os.makedirs(video_output_dir, exist_ok=True)
        extract_and_save_frames(video_path, video_output_dir)

    for video_name in val_videos:
        video_path = os.path.join(video_root, video_name)
        video_output_dir = os.path.join(word_val_dir, os.path.splitext(video_name)[0])

        if os.path.exists(os.path.join(video_output_dir, "frame_0000.jpg")):
            print(f"⏩ Skipping {video_name} (already extracted)")
            continue

        if not os.path.exists(video_path):
            print(f"❌ Missing video: {video_path}")
            continue

        os.makedirs(video_output_dir, exist_ok=True)
        extract_and_save_frames(video_path, video_output_dir)

print("✅ Frame extraction and splitting complete!")
