import os
import shutil
import random
from tqdm import tqdm

# Dataset names (choose one at a time)
dataset_name = "20_words"  # Change to "50_words", "100_words", or "200_words" as needed

# Paths
frames_dir = f"~/Projects/HandiSpeakV2/frames/{dataset_name}"
train_dir = f"~/Projects/HandiSpeakV2/datasets/{dataset_name}/train"
val_dir = f"~/Projects/HandiSpeakV2/datasets/{dataset_name}/val"

# Create train and validation directories
os.makedirs(os.path.expanduser(train_dir), exist_ok=True)
os.makedirs(os.path.expanduser(val_dir), exist_ok=True)

# Train/Val split ratio
train_ratio = 0.8

# Split data
for word in tqdm(os.listdir(os.path.expanduser(frames_dir)), desc="Splitting data"):
    print(f"Processing word: {word}")
    word_path = os.path.expanduser(os.path.join(frames_dir, word))
    if not os.path.isdir(word_path):
        continue

    # Get list of all video frame folders for this word
    video_folders = [f for f in os.listdir(word_path) if os.path.isdir(os.path.join(word_path, f))]

    # Shuffle the list for random splitting
    random.shuffle(video_folders)

    # Calculate the split point
    split_point = int(len(video_folders) * train_ratio)

    # Split into train and validation
    train_videos = video_folders[:split_point]
    val_videos = video_folders[split_point:]

    # Move training frames
    for video in train_videos:
        video_src = os.path.join(word_path, video)
        video_dst = os.path.join(os.path.expanduser(train_dir), word, video)
        os.makedirs(os.path.dirname(video_dst), exist_ok=True)

        # Move each frame inside the video folder
        for frame in os.listdir(video_src):
            src_frame_path = os.path.join(video_src, frame)
            dst_frame_path = os.path.join(video_dst, frame)
            os.makedirs(os.path.dirname(dst_frame_path), exist_ok=True)
            shutil.move(src_frame_path, dst_frame_path)

        # Remove empty folder after moving frames
        os.rmdir(video_src)

    # Move validation frames
    for video in val_videos:
        video_src = os.path.join(word_path, video)
        video_dst = os.path.join(os.path.expanduser(val_dir), word, video)
        os.makedirs(os.path.dirname(video_dst), exist_ok=True)

        # Move each frame inside the video folder
        for frame in os.listdir(video_src):
            src_frame_path = os.path.join(video_src, frame)
            dst_frame_path = os.path.join(video_dst, frame)
            os.makedirs(os.path.dirname(dst_frame_path), exist_ok=True)
            shutil.move(src_frame_path, dst_frame_path)

        # Remove empty folder after moving frames
        os.rmdir(video_src)

print("✅ Training and validation data split complete.")
