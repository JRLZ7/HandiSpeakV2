import cv2
import mediapipe as mp
import os
import json
from tqdm import tqdm

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Keypoint indices for face and pose
FACE_LANDMARKS = [
    70, 63, 105, 66, 107, 46, 53, 52, 65, 55,  # Left eyebrow
    336, 296, 334, 293, 300, 285, 295, 282, 283, 276,  # Right eyebrow
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,  # Left eye
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,  # Right eye
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,  # Upper lip
    95, 88, 178, 87, 14, 317, 402, 318, 324  # Lower lip

    # 468, 473 removed since they aren't picked up in any json files.
]
POSE_LANDMARKS = [11, 12, 13, 14]  # Upper torso and arms
NUM_FRAMES = 20

def round_coord(coord):
    return round(coord, 5)

def extract_keypoints(image, results):
    keypoints = {}
    missing = {}

    # FACE
    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark
        keypoints['face'] = [
            {
                'id': i,
                'x': round_coord(face_landmarks[i].x),
                'y': round_coord(face_landmarks[i].y),
                'z': round_coord(face_landmarks[i].z)
            }
            for i in FACE_LANDMARKS if i < len(face_landmarks)
        ]
    else:
        keypoints['face'] = [
            {'id': i, 'x': 0.0, 'y': 0.0, 'z': 0.0}
            for i in FACE_LANDMARKS
        ]


    # POSE
    if results.pose_landmarks:
        keypoints['pose'] = [{'id': i,
                              'x': round_coord(results.pose_landmarks.landmark[i].x),
                              'y': round_coord(results.pose_landmarks.landmark[i].y),
                              'z': round_coord(results.pose_landmarks.landmark[i].z)}
                             for i in POSE_LANDMARKS]
        missing['pose'] = False
    else:
        keypoints['pose'] = [{'id': i, 'x': 0.0, 'y': 0.0, 'z': 0.0} for i in POSE_LANDMARKS]
        missing['pose'] = True

    # LEFT HAND
    if results.left_hand_landmarks:
        keypoints['left_hand'] = [{'id': idx,
                                   'x': round_coord(lm.x),
                                   'y': round_coord(lm.y),
                                   'z': round_coord(lm.z)}
                                  for idx, lm in enumerate(results.left_hand_landmarks.landmark)]
        missing['left_hand'] = False
    else:
        keypoints['left_hand'] = [{'id': i, 'x': 0.0, 'y': 0.0, 'z': 0.0} for i in range(21)]
        missing['left_hand'] = True

    # RIGHT HAND
    if results.right_hand_landmarks:
        keypoints['right_hand'] = [{'id': idx,
                                    'x': round_coord(lm.x),
                                    'y': round_coord(lm.y),
                                    'z': round_coord(lm.z)}
                                   for idx, lm in enumerate(results.right_hand_landmarks.landmark)]
        missing['right_hand'] = False
    else:
        keypoints['right_hand'] = [{'id': i, 'x': 0.0, 'y': 0.0, 'z': 0.0} for i in range(21)]
        missing['right_hand'] = True

    keypoints['missing'] = missing
    return keypoints

# Save directory
os.makedirs('keypoints', exist_ok=True)

# Load metadata
metadata_path = '/home/jason/Projects/HandiSpeakV2/datasets/top_200_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

batch_keypoints = {}

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for word, instances in tqdm(metadata.items(), desc="Extracting keypoints"):

        json_path = os.path.join('keypoints', f'{word}.json')
        if os.path.exists(json_path):
            print(f"⏩ Skipping {word} (already processed)")
            continue
        
        word_keypoints = {}
        for instance in instances:
            video_path = f"data/WLASL/videos/{instance}"
            if not os.path.exists(video_path):
                print(f"❌ Video not found: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Cannot open video: {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, frame_count // NUM_FRAMES)

            video_keypoints = []
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)
                    keypoints = extract_keypoints(image, results)
                    video_keypoints.append(keypoints)
                frame_idx += 1

            cap.release()
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            word_keypoints[video_id] = video_keypoints

        with open(json_path, 'w') as f:
            json.dump(word_keypoints, f)

print("✅ Batch keypoint extraction complete!")
