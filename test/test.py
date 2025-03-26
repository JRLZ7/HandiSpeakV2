import os
import json

data_dir = "/home/jason/Projects/HandiSpeakV2/keypoints"  # Adjust the path as needed

for file_name in os.listdir(data_dir):
    if file_name.endswith(".json"):
        file_path = os.path.join(data_dir, file_name)
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Check if the outer structure is a dictionary
            if not isinstance(data, dict):
                print(f"🚨 Problematic JSON structure (not a dict) in: {file_name}")
                continue
            
            # Check each video ID inside the JSON
            for video_id, video_data in data.items():
                if not isinstance(video_data, list):
                    print(f"🚨 Video data for '{video_id}' in '{file_name}' is not a list, but {type(video_data)}")
                
                # Check each frame in the video data
                for frame in video_data:
                    if not isinstance(frame, dict):
                        print(f"🚨 Frame data in video '{video_id}' from '{file_name}' is not a dict but {type(frame)}")
                    else:
                        for part, keypoints in frame.items():
                            if not isinstance(keypoints, list):
                                print(f"🚨 Keypoints for '{part}' in frame of video '{video_id}' from '{file_name}' is not a list, but {type(keypoints)}")

        except Exception as e:
            print(f"🚨 Error reading {file_name}: {e}")
