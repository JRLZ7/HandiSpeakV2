import os
import json
import random
import numpy as np
from tqdm import tqdm
import time

# Augmentation parameters
NUM_AUGMENTED_VIDEOS = 50  # Total videos per word after augmentation
TRANSLATION_FACTOR = 0.02  # Factor for translating keypoints
NOISE_STD_DEV = 0.005  # Standard deviation for Gaussian noise

# Paths
ORIGINAL_DIR = "keypoints"  # Original keypoints directory
AUGMENTED_DIR = "keypoints_aug"  # Augmented keypoints directory

os.makedirs(AUGMENTED_DIR, exist_ok=True)

def add_noise(keypoints):
    noisy_keypoints = []
    noise = np.random.normal(0, NOISE_STD_DEV, (len(keypoints), 3))  # Generate noise for each keypoint
    for i, kp in enumerate(keypoints):
        # Check if keypoint is zeroed (indicating missing data), skip if true
        if kp['x'] == 0.0 and kp['y'] == 0.0 and kp['z'] == 0.0:
            noisy_keypoints.append(kp)
            continue

        # Apply noise without clamping z-coordinates
        noisy_kp = {
            'x': min(max(kp['x'] + noise[i, 0], 0.0), 1.0),
            'y': min(max(kp['y'] + noise[i, 1], 0.0), 1.0),
            'z': kp['z'] + noise[i, 2]  # No clamping for Z
        }
        noisy_keypoints.append(noisy_kp)
    return noisy_keypoints

def translate(keypoints, dx, dy):
    translated = []
    for point in keypoints:
        # Check if keypoint is zeroed (indicating missing data), skip if true
        if point['x'] == 0.0 and point['y'] == 0.0 and point['z'] == 0.0:
            translated.append(point)
            continue

        # Apply translation without modifying Z
        translated_point = {
            'x': min(max(point['x'] + dx, 0.0), 1.0),
            'y': min(max(point['y'] + dy, 0.0), 1.0),
            'z': point['z']  # Preserve Z
        }
        translated.append(translated_point)
    return translated

def augment_video(video):
    augmented_video = []
    for frame in video:
        augmented_frame = {}
        for part, keypoints in frame.items():
            # Skip augmentation for zeroed frames
            if not keypoints:
                augmented_frame[part] = keypoints
                continue

            # Choose augmentation: noise or translation
            if random.random() < 0.5:
                augmented_frame[part] = add_noise(keypoints)
            else:
                dx = random.uniform(-TRANSLATION_FACTOR, TRANSLATION_FACTOR)
                dy = random.uniform(-TRANSLATION_FACTOR, TRANSLATION_FACTOR)
                augmented_frame[part] = translate(keypoints, dx, dy)

        augmented_video.append(augmented_frame)
    return augmented_video

def augment_word(word, original_videos):
    video_list = list(original_videos.items())  # Convert to list of (video_id, data)
    augmented_videos = {}

    print(f"[INFO] Augmenting '{word}' (has {len(video_list)} videos), aiming for {NUM_AUGMENTED_VIDEOS} total.")

    for video_id, video_data in video_list:
        augmented_videos[video_id] = video_data  # Include original video

    num_augmentations = NUM_AUGMENTED_VIDEOS - len(video_list)

    for i in range(num_augmentations):
        original_video_id, video = random.choice(video_list)  # Pick a random video to augment
        augmented_video = augment_video(video)
        aug_video_id = f"aug_{i+1}_{original_video_id}"
        augmented_videos[aug_video_id] = augmented_video  # Store augmented video with new ID

        # Print progress
        if (i + 1) % 10 == 0 or i == num_augmentations - 1:
            print(f"  [INFO] Augmentation progress for '{word}': {i + 1}/{num_augmentations} videos")

    print(f"[INFO] Augmentation complete for '{word}' with {len(augmented_videos)} total videos.")
    return augmented_videos

def augment_keypoints():
    print("Starting augmentation...")

    # Iterate through original keypoint files
    for filename in tqdm(os.listdir(ORIGINAL_DIR), desc="Processing words"):
        if filename.endswith(".json"):
            word = filename.replace(".json", "")
            original_path = os.path.join(ORIGINAL_DIR, filename)

            try:
                # Load original keypoints
                with open(original_path, "r") as f:
                    original_videos = json.load(f)

                # Check if the loaded content is a dictionary
                if not isinstance(original_videos, dict):
                    print(f"🚨 Error: Unexpected data format in {original_path}. Skipping.")
                    continue

                # Perform augmentation
                start_time = time.time()
                augmented_videos = augment_word(word, original_videos)
                end_time = time.time()
                duration = end_time - start_time
                print(f"[INFO] Augmentation for '{word}' took {duration:.2f} seconds.")

                # Save augmented keypoints
                augmented_path = os.path.join(AUGMENTED_DIR, f"{word}.json")
                with open(augmented_path, "w") as f:
                    json.dump(augmented_videos, f)

                print(f"[INFO] Saved augmented data for '{word}' to {augmented_path}")

            except Exception as e:
                print(f"🚨 Error processing '{filename}': {e}")

    print("✅ Augmentation complete for all words!")

if __name__ == "__main__":
    augment_keypoints()
