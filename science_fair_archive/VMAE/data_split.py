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
train_ratio = 0.8  # Train/validation split ratio

# Dataset names (choose one at a time)
dataset_name = "top_50_metadata"  # Change to "50_words", "100_words", or "200_words" as needed

# Paths
metadata_path = f"~/Projects/HandiSpeakV2/datasets/{dataset_name}_metadata.json"
train_dir = f"~/Projects/HandiSpeakV2/datasets/{dataset_name}/train"
val_dir = f"~/Projects/HandiSpeakV2/datasets/{dataset_name}/val"

# Create train and validation directories
os.makedirs(os.path.expanduser(train_dir), exist_ok=True)
os.makedirs(os.path.expanduser(val_dir), exist_ok=True)

# Load metadata
with open(os.path.expanduser(metadata_path), "r") as file:
    metadata = json.load(file)

# Function to normalize frames to a fixed length
def normalize_frames(frames):
    frame_count = len(frames)

    # Truncate or pad frames to match the desired count
    if frame_count > desired_frames:
        indices = np.linspace(0, frame_count - 1, desired_frames).astype(int)
        frames = [frames[i] for i in indices]
    elif frame_count < desired_frames:
        # Pad with the last frame
        pad_count = desired_frames - frame_count
        frames.extend([frames[-1]] * pad_count)

    return frames

# Frame extraction function
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

        # Only process frames at the calculated interval
        if count % interval == 0:
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)

        count += 1

    cap.release()

    # Normalize the frames to ensure consistency
    frames = normalize_frames(frames)

    # Save normalized frames to the output directory
    for i, frame in enumerate(frames):
        frame_filename = f"frame_{i:04d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

# Process each word and video
for word, videos in tqdm(metadata.items(), desc="Splitting and Extracting Data"):
    word_train_dir = os.path.expanduser(os.path.join(train_dir, word))
    word_val_dir = os.path.expanduser(os.path.join(val_dir, word))
    os.makedirs(word_train_dir, exist_ok=True)
    os.makedirs(word_val_dir, exist_ok=True)

    # Shuffle video list for random split
    random.shuffle(videos)
    split_point = int(len(videos) * train_ratio)
    train_videos = videos[:split_point]
    val_videos = videos[split_point:]

    # Process training videos
    for video_name in train_videos:
        video_path = os.path.expanduser(f"~/Projects/HandiSpeakV2/data/WLASL/videos/{video_name}")
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            continue

        # Create video-specific directory within the word folder
        video_output_dir = os.path.join(word_train_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        extract_and_save_frames(video_path, video_output_dir)

    # Process validation videos
    for video_name in val_videos:
        video_path = os.path.expanduser(f"~/Projects/HandiSpeakV2/data/WLASL/videos/{video_name}")
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            continue

        # Create video-specific directory within the word folder
        video_output_dir = os.path.join(word_val_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        extract_and_save_frames(video_path, video_output_dir)

print("✅ Frame extraction and splitting complete!")
