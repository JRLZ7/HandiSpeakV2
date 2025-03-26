import cv2
import json
import os

# Load the JSON file
json_path = '/home/jason/Projects/HandiSpeakV2/keypoints/go.json'  # Update this to your JSON file path
with open(json_path, 'r') as f:
    keypoints_data = json.load(f)

# Get video path from JSON metadata (assuming consistent naming)
video_id = list(keypoints_data.keys())[0]  # Take the first key as video ID
video_path = f'data/WLASL/videos/{video_id}.mp4'

# Open the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Check if the current frame index exists in the keypoints data
    if frame_index < len(keypoints_data[video_id]):
        keypoints = keypoints_data[video_id][frame_index]

        # Draw left hand keypoints
        if 'left_hand' in keypoints:
            for point in keypoints['left_hand']:
                x = int(point['x'] * frame.shape[1])
                y = int(point['y'] * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # Draw right hand keypoints
        if 'right_hand' in keypoints:
            for point in keypoints['right_hand']:
                x = int(point['x'] * frame.shape[1])
                y = int(point['y'] * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Draw face keypoints
        if 'face' in keypoints:
            for point in keypoints['face']:
                x = int(point['x'] * frame.shape[1])
                y = int(point['y'] * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        # Draw pose keypoints
        if 'pose' in keypoints:
            for point in keypoints['pose']:
                x = int(point['x'] * frame.shape[1])
                y = int(point['y'] * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (255, 255, 0), -1)

    # Display the frame with overlaid keypoints
    cv2.imshow('Keypoint Visualization', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()
