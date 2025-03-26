import os
import json
import subprocess

# Problematic video IDs and their corresponding files
problematic_videos = {
    "trade.json": ["59212"],
    "help.json": ["27209"],
    "before.json": ["05734"],
    "what.json": ["62987"],
    "who.json": ["63231", "63232"],
    "cool.json": ["13202"],
    "drink.json": ["17733", "17734"],
    "thin.json": ["57942", "57947"],
    "cousin.json": ["13647", "13648", "13635"]
}

# Path to the original keypoints directory
data_dir = "/home/jason/Projects/HandiSpeakV2/keypoints"

def reprocess_video(video_id, file_name):
    json_path = os.path.join(data_dir, file_name)

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        print(f"Reprocessing video ID {video_id} in file {file_name}...")

        # Check if the video ID exists in the JSON file
        if video_id not in data:
            print(f"Video ID {video_id} not found in {file_name}. Skipping.")
            return

        # Update the keypoints to match the consistent format (354 keypoints)
        reprocessed_data = {}
        for frame_idx, frame in enumerate(data[video_id]):
            frame_keypoints = []

            for part in ['face', 'pose', 'left_hand', 'right_hand']:
                if part in frame:
                    keypoints = frame[part]
                    frame_keypoints.extend([kp['x'] for kp in keypoints])
                    frame_keypoints.extend([kp['y'] for kp in keypoints])
                    frame_keypoints.extend([kp['z'] for kp in keypoints])
                else:
                    # Fill with zeroed keypoints if part is missing
                    if part == 'face':
                        num_keypoints = 71
                    elif part == 'pose':
                        num_keypoints = 4
                    else:  # left_hand or right_hand
                        num_keypoints = 21

                    frame_keypoints.extend([0.0] * 3 * num_keypoints)

            # Store the updated frame
            reprocessed_data[frame_idx] = {
                "keypoints": frame_keypoints,
                "length": len(frame_keypoints)
            }

        # Replace the old video data with the reprocessed data
        data[video_id] = reprocessed_data

        # Save the updated JSON file
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"✅ Successfully reprocessed video ID {video_id} in file {file_name}")

    except Exception as e:
        print(f"Error reprocessing video ID {video_id} in file {file_name}: {e}")

# Reprocess each problematic video
for file_name, video_ids in problematic_videos.items():
    for video_id in video_ids:
        reprocess_video(video_id, file_name)

print("🎉 Reprocessing complete!")
