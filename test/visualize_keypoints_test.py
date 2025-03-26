import cv2
import json
import os
from tqdm import tqdm
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def visualize_keypoints(json_path, video_dir):
    # Load keypoints JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Iterate through each video ID in the JSON
    for video_id in tqdm(data.keys(), desc="Visualizing keypoints"):
        # Determine the video file path
        if "aug" in video_id:
            original_id = video_id.split("_")[-1]  # Extract the original video ID from augmented name
            video_file = f"{original_id}.mp4"
        else:
            original_id = video_id
            video_file = f"{video_id}.mp4"

        video_path = os.path.join(video_dir, video_file)

        # Check if the video file exists
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            continue

        print(f"Visualizing video: {video_id} (Original ID: {original_id})")

        cap = cv2.VideoCapture(video_path)

        # Extract frame indices
        frame_indices = list(range(len(data[video_id])))

        for frame_number in frame_indices:
            ret, frame = cap.read()
            if not ret:
                print(f"[ERROR] Unable to read frame {frame_number} from {video_path}")
                break

            # Retrieve keypoints for the current frame
            frame_keypoints = data[video_id][frame_number]

            # Draw face keypoints
            if 'face' in frame_keypoints:
                for kp in frame_keypoints['face']:
                    x, y = int(kp['x'] * frame.shape[1]), int(kp['y'] * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Draw pose keypoints
            if 'pose' in frame_keypoints:
                for kp in frame_keypoints['pose']:
                    x, y = int(kp['x'] * frame.shape[1]), int(kp['y'] * frame.shape[0])
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

            # Draw left hand keypoints
            if 'left_hand' in frame_keypoints:
                for kp in frame_keypoints['left_hand']:
                    x, y = int(kp['x'] * frame.shape[1]), int(kp['y'] * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            # Draw right hand keypoints
            if 'right_hand' in frame_keypoints:
                for kp in frame_keypoints['right_hand']:
                    x, y = int(kp['x'] * frame.shape[1]), int(kp['y'] * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

            # Display the frame
            cv2.imshow(f"Video {video_id}", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):  # Wait for a key press
                break

        cap.release()
        cv2.destroyAllWindows()

    print("✅ Visualization complete!")

# Usage example
json_path = "keypoints/before.json"  # Update with your file path
video_dir = "data/WLASL/videos"  # Update with your video directory
visualize_keypoints(json_path, video_dir)
