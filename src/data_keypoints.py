import cv2
import mediapipe as mp
import os
import json
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Function to extract and save keypoints
def extract_keypoints(image, results):
    keypoints = {}

    if results.left_hand_landmarks:
        keypoints['left_hand'] = [
            {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
            for landmark in results.left_hand_landmarks.landmark
        ]

    if results.right_hand_landmarks:
        keypoints['right_hand'] = [
            {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
            for landmark in results.right_hand_landmarks.landmark
        ]

    if results.pose_landmarks:
        keypoints['pose'] = [
            {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
            for landmark in results.pose_landmarks.landmark
        ]

    return keypoints

# Directory setup
os.makedirs('keypoints', exist_ok=True)

# Load video file paths from 20_words_metadata.json
metadata_path = '/home/jason/Projects/HandiSpeakV2/datasets/20_words_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Batch keypoints by word
batch_keypoints = {}

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for word, instances in tqdm(metadata.items(), desc="Extracting keypoints"):
        print(f"Processing word: {word}")
        word_videos = []

        for instance in instances:
            video_id = str(instance)  # Use the instance name (video ID) as identifier
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

            # Append keypoints for this video to the word's batch
            word_videos.append({
                "video_id": video_id,
                "keypoints": video_keypoints
            })

        # Save the batched keypoints as one JSON file per word
        batch_keypoints[word] = {"word": word, "videos": word_videos}
        json_path = os.path.join('keypoints', f'{word}.json')
        print(f"Saving keypoints to {json_path}")
        with open(json_path, 'w') as f:
            json.dump(batch_keypoints[word], f)

        print(f"✅ Completed processing word: {word}")

print("✅ Batch keypoint extraction complete!")
