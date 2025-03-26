import cv2
import mediapipe as mp
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Keypoint indices for face and pose
FACE_LANDMARKS = [
    70, 63, 105, 66, 107, 46, 53, 52, 65, 55,  # Left eyebrow
    336, 296, 334, 293, 300, 285, 295, 282, 283, 276,  # Right eyebrow
    468, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,  # Left eye
    473, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,  # Right eye
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,  # Upper lip
    95, 88, 178, 87, 14, 317, 402, 318, 324  # Lower lip
]
POSE_LANDMARKS = [11, 12, 13, 14]  # Upper torso and arms

# Frame interval configuration
NUM_FRAMES = 20

# Extract keypoints from MediaPipe results
def extract_keypoints(image, results):
    keypoints = {}

    # Face keypoints
    if results.face_landmarks:
        keypoints['face'] = [
            {'x': results.face_landmarks.landmark[i].x,
             'y': results.face_landmarks.landmark[i].y,
             'z': results.face_landmarks.landmark[i].z}
            for i in FACE_LANDMARKS if i < len(results.face_landmarks.landmark)
        ]
    else:
        keypoints['face'] = [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(len(FACE_LANDMARKS))]

    # Pose keypoints
    if results.pose_landmarks:
        keypoints['pose'] = [
            {'x': results.pose_landmarks.landmark[i].x,
             'y': results.pose_landmarks.landmark[i].y,
             'z': results.pose_landmarks.landmark[i].z}
            for i in POSE_LANDMARKS if i < len(results.pose_landmarks.landmark)
        ]
    else:
        keypoints['pose'] = [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(len(POSE_LANDMARKS))]

    # Left hand keypoints
    if results.left_hand_landmarks:
        keypoints['left_hand'] = [
            {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
            for landmark in results.left_hand_landmarks.landmark
        ]
    else:
        keypoints['left_hand'] = [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(21)]

    # Right hand keypoints
    if results.right_hand_landmarks:
        keypoints['right_hand'] = [
            {'x': landmark.x, 'y': landmark.y, 'z': landmark.z}
            for landmark in results.right_hand_landmarks.landmark
        ]
    else:
        keypoints['right_hand'] = [{'x': 0.0, 'y': 0.0, 'z': 0.0} for _ in range(21)]

    return keypoints

# Visualize the keypoints on a 2D plot
def visualize_keypoints(keypoints):
    """
    Visualizes the keypoints on a 2D plot to inspect their positions.
    """
    # Extract the x, y coordinates for each part (face, pose, left hand, right hand)
    for part in ['face', 'pose', 'left_hand', 'right_hand']:
        points = keypoints.get(part, [])
        if points:
            x = [kp['x'] for kp in points]
            y = [kp['y'] for kp in points]
            plt.scatter(x, y, label=part, s=10)  # Draw points on the plot

    # Set plot limits to prevent points from getting out of bounds
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Visualized Keypoints")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()

# Create keypoints directory
os.makedirs('keypoints', exist_ok=True)

# Load metadata
metadata_path = '/home/jason/Projects/HandiSpeakV2/datasets/top_20_metadata.json'  # Replace with your actual metadata file
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

batch_keypoints = {}

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for word, instances in tqdm(metadata.items(), desc="Extracting keypoints"):
        word_keypoints = {}
        for instance in instances:
            video_path = f"data/WLASL/videos/{instance}"

            if not os.path.exists(video_path):
                print(f"Error: Could not find video file {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_path}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, frame_count // NUM_FRAMES)

            video_keypoints = []

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process only every frame_interval frame
                if frame_idx % frame_interval == 0:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = holistic.process(image)

                    # Extract and store keypoints
                    keypoints = extract_keypoints(image, results)

                    # Visualize the keypoints on every frame
                    visualize_keypoints(keypoints)

                    video_keypoints.append(keypoints)

                frame_idx += 1

            cap.release()

            # Add keypoints to the word dictionary
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            word_keypoints[video_id] = video_keypoints

        # Save the batched keypoints as one JSON file per word
        json_path = os.path.join('keypoints', f'{word}.json')
        with open(json_path, 'w') as f:
            json.dump(word_keypoints, f)

print("✅ Batch keypoint extraction complete!")
