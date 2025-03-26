import os
import json
import subprocess

# Directory paths
keypoints_dir = "/home/jason/Projects/HandiSpeakV2/keypoints"  # Adjust if needed
video_dir = "/home/jason/Projects/HandiSpeakV2/data/WLASL/videos"  # Path to the original video files

def delete_problematic_files():
    print("🚨 Deleting problematic files...")

    for file_name in os.listdir(keypoints_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(keypoints_dir, file_name)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                for video_id, video_data in data.items():
                    # Check for problematic structure
                    if not isinstance(video_data, list) or any(not isinstance(frame, dict) for frame in video_data):
                        print(f"🚨 Deleting corrupted file: {file_name} (problem with video ID: {video_id})")
                        os.remove(file_path)
                        break

            except Exception as e:
                print(f"🚨 Failed to read {file_name}: {e}")

    print("✅ Problematic files deleted.")

def reextract_keypoints():
    print("🚀 Re-extracting keypoints from videos...")

    # Use the original keypoint extraction command for each video
    for video_name in os.listdir(video_dir):
        if video_name.endswith(".mp4"):  # Adjust video extension if necessary
            video_path = os.path.join(video_dir, video_name)
            output_json = os.path.join(keypoints_dir, f"{os.path.splitext(video_name)[0]}.json")

            print(f"🔄 Re-extracting keypoints for: {video_name}")
            command = f"python3 test/data_keypoints_test.py --video {video_path} --output {output_json}"
            result = subprocess.run(command, shell=True, capture_output=True)

            if result.returncode != 0:
                print(f"❌ Failed to extract keypoints from {video_name}")
                print(result.stderr.decode())
            else:
                print(f"✅ Successfully extracted keypoints for: {video_name}")

    print("✅ Keypoint re-extraction complete.")

if __name__ == "__main__":
    delete_problematic_files()
    reextract_keypoints()
