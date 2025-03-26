import os
import json

def check_keypoint_lengths(data_dir):
    print(f"Checking keypoint lengths in directory: {data_dir}")
    keypoint_counts = {354: 0, 360: 0, "other": 0}

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(data_dir, file_name)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for video_id, video_data in data.items():
                        for frame in video_data:
                            keypoint_length = 0
                            for part in ['face', 'pose', 'left_hand', 'right_hand']:
                                if part in frame:
                                    keypoints = frame[part]
                                    keypoint_length += len(keypoints) * 3
                                else:
                                    if part == 'face':
                                        keypoint_length += 71 * 3
                                    elif part == 'pose':
                                        keypoint_length += 4 * 3
                                    else:
                                        keypoint_length += 21 * 3

                            if keypoint_length == 360:
                                print(f"🚨 Problematic Original Video: {video_id} in file {file_name} has {keypoint_length} keypoints")
                            if keypoint_length == 354:
                                keypoint_counts[354] += 1
                            elif keypoint_length == 360:
                                keypoint_counts[360] += 1
                            else:
                                keypoint_counts["other"] += 1
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    print("\nSummary of Keypoint Lengths in Original Data:")
    print(f"354 Keypoints: {keypoint_counts[354]}")
    print(f"360 Keypoints: {keypoint_counts[360]}")
    print(f"Other Keypoints: {keypoint_counts['other']}")

# Run the script for the original data directory
check_keypoint_lengths('/home/jason/Projects/HandiSpeakV2/keypoints')
