import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from pathlib import Path
import random

# Setup
video_path = "/home/jason/Projects/HandiSpeakV2/data/WLASL/videos/69345.mp4"
output_dir = Path("photos")
output_dir.mkdir(exist_ok=True)

# Step 1: Extract middle frame
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
mid_frame_index = frame_count // 2
target_frame = None

for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    if i == mid_frame_index:
        target_frame = frame
        break
cap.release()

# Save raw frame
cv2.imwrite(str(output_dir / "raw_frame.png"), target_frame)

# Step 2: Extract keypoints
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

frame_rgb = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

with mp_holistic.Holistic(static_image_mode=True) as holistic:
    results = holistic.process(frame_rgb)

def extract_coords(results):
    keypoints = {}
    FACE_LANDMARKS = [
    70, 63, 105, 66, 107, 46, 53, 52, 65, 55,  # Left eyebrow
    336, 296, 334, 293, 300, 285, 295, 282, 283, 276,  # Right eyebrow
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,  # Left eye
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,  # Right eye
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,  # Upper lip
    95, 88, 178, 87, 14, 317, 402, 318, 324  # Lower lip
]
    POSE_LANDMARKS = [11, 12, 13, 14]

    if results.face_landmarks:
        keypoints['face'] = [
            (results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y)
            for i in FACE_LANDMARKS if i < len(results.face_landmarks.landmark)
        ]

    if results.pose_landmarks:
        keypoints['pose'] = [
            (results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y)
            for i in POSE_LANDMARKS if i < len(results.pose_landmarks.landmark)
        ]

    if results.left_hand_landmarks:
        keypoints['left_hand'] = [(lm.x, lm.y) for lm in results.left_hand_landmarks.landmark]
    if results.right_hand_landmarks:
        keypoints['right_hand'] = [(lm.x, lm.y) for lm in results.right_hand_landmarks.landmark]
    return keypoints

original_keypoints = extract_coords(results)

# Step 3: Plot function
def plot_keypoints(kps, title, filename):
    plt.figure(figsize=(5, 5))
    for part, coords in kps.items():
        x = [p[0] for p in coords]
        y = [1 - p[1] for p in coords]  # Flip y for display
        plt.scatter(x, y, label=part, s=10)
    plt.legend()
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()

plot_keypoints(original_keypoints, "Original Keypoints", "original_keypoints.png")

# Step 4: Apply Gaussian noise
def add_noise(kps, level=0.01):
    noisy_kps = {}
    for part, coords in kps.items():
        noisy_kps[part] = [(x + random.uniform(-level, level),
                            y + random.uniform(-level, level)) for x, y in coords]
    return noisy_kps

keypoints_noise = add_noise(original_keypoints)
plot_keypoints(keypoints_noise, "Gaussian Noise", "keypoints_noise.png")

# Step 5: Apply horizontal translation
def translate_x(kps, offset=0.05):
    translated_kps = {}
    for part, coords in kps.items():
        translated_kps[part] = [(x + offset, y) for x, y in coords]
    return translated_kps

keypoints_translated = translate_x(original_keypoints)
plot_keypoints(keypoints_translated, "Horizontal Translation", "keypoints_translate.png")
