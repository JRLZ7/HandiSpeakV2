import cv2
import mediapipe as mp
import os
import json
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Define the keypoints to extract for face (eyebrows, eyes, lips)
FACE_LANDMARKS = [
    # Left eyebrow
    70, 63, 105, 66, 107, 46, 53, 52, 65, 55,
    # Right eyebrow
    336, 296, 334, 293, 300, 285, 295, 282, 283, 276,
    # Left eye
    468, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    # Right eye
    473, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    # Upper lip
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    # Lower lip
    95, 88, 178, 87, 14, 317, 402, 318, 324
]

# Define the keypoints to extract for pose (upper chest and above)
POSE_LANDMARKS = [
    # Upper torso and arms
    11, 12, 13, 14
]

# Function to extract and save selected keypoints
def extract_keypoints(image, results):
    keypoints = {}

    # Left hand landmarks (all)
    if results.left_hand_landmarks:
        keypoints['left_hand'] = [
            {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
            for landmark in results.left_hand_landmarks.landmark
        ]

    # Right hand landmarks (all)
    if results.right_hand_landmarks:
        keypoints['right_hand'] = [
            {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
            for landmark in results.right_hand_landmarks.landmark
        ]

    # Upper chest up pose landmarks (including shoulders, arms)
    if results.pose_landmarks:
        keypoints['pose'] = [
            {'x': results.pose_landmarks.landmark[i].x,
             'y': results.pose_landmarks.landmark[i].y,
             'z': results.pose_landmarks.landmark[i].z}
            for i in POSE_LANDMARKS
        ]

    # Face landmarks (eyebrows, eyes, lips)
    if results.face_landmarks:
        keypoints['face'] = [
            {'x': results.face_landmarks.landmark[i].x,
             'y': results.face_landmarks.landmark[i].y,
             'z': results.face_landmarks.landmark[i].z}
            for i in FACE_LANDMARKS
        ]

    return keypoints

# Directory setup
os.makedirs('keypoints', exist_ok=True)

# Load video file paths from 20_words_metadata.json
metadata_path = '/home/jason/Projects/HandiSpeakV2/datasets/20_words_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Extract and save keypoints for each word
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for word, instances in tqdm(metadata.items(), desc="Extracting keypoints"):
        print(f"Processing word: {word}")
        word_keypoints = {}

        for instance in instances:
            video_id = str(instance)
            video_path = f"data/WLASL/videos/{video_id}"
            print(f"Attempting to open video file: {video_path}")

            if not os.path.exists(video_path):
                print(f"Error: Could not find video file {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_path}")
                continue

            video_keypoints = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                keypoints = extract_keypoints(image, results)
                video_keypoints.append(keypoints)

            cap.release()

            # Add the keypoints for this video to the word dictionary
            word_keypoints[video_id] = video_keypoints
            print(f"Completed processing video: {video_id}")

        # Save keypoints for the current word
        json_path = os.path.join('keypoints', f'{word}.json')
        print(f"Saving keypoints to {json_path}")
        with open(json_path, 'w') as f:
            json.dump(word_keypoints, f)

        print(f"Completed processing word: {word}")

print("✅ Batch keypoint extraction complete!")