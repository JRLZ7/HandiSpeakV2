import cv2
import os
import json
import numpy as np
from tqdm import tqdm

# Parameters
desired_fps = 30
desired_frames = 60
frame_size = (224, 224)

# Dataset names (choose one at a time)
dataset_name = "20_words"  # Change to "50_words", "100_words", or "200_words" as needed

# Paths
metadata_path = f"~/Projects/HandiSpeakV2/datasets/{dataset_name}_metadata.json"
output_dir = f"~/Projects/HandiSpeakV2/frames/{dataset_name}"
os.makedirs(os.path.expanduser(output_dir), exist_ok=True)

# Load metadata
with open(os.path.expanduser(metadata_path), "r") as file:
    metadata = json.load(file)

# Function to normalize the number of frames
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
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate interval to achieve desired FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(frame_count / desired_frames))

    frames = []
    count = 0

    print(f"Video: {video_path} | FPS: {fps} | Total frames: {frame_count} | Interval: {interval}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only save frames at the calculated interval
        if count % interval == 0:
            frame = cv2.resize(frame, frame_size)  # Resize to 224x224
            frames.append(frame)
        
        count += 1

    cap.release()

    # Ensure we have exactly 60 frames (pad or sample if necessary)
    frames = normalize_frames(frames)
    return frames

# def extract_frames(video_path):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     interval = max(1, int(fps / desired_fps))  # Normalize to desired FPS

#     frames = []
#     count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Only process frames at the desired interval
#         if count % interval == 0:
#             frame = cv2.resize(frame, frame_size)  # Resize to 224x224
#             frames.append(frame)

#         count += 1

#     cap.release()
#     return normalize_frames(frames)

# Process each word and video
for word, videos in tqdm(metadata.items(), desc="Processing videos"):
    word_dir = os.path.expanduser(f"{output_dir}/{word}")
    os.makedirs(word_dir, exist_ok=True)

    for video_name in videos:
        video_path = os.path.expanduser(f"~/Projects/HandiSpeakV2/data/WLASL/videos/{video_name}")
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            continue

        # Extract and normalize frames
        frames = extract_frames(video_path)

        # Save frames
        for i, frame in enumerate(frames):
            output_frame_path = f"{word_dir}/{video_name}_frame_{i:04d}.jpg"
            cv2.imwrite(os.path.expanduser(output_frame_path), frame)

        print(f"✅ Processed {video_name} for word '{word}'")

print("✅ Frame extraction and normalization complete.")
