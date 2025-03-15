import os
import json
import cv2
from pathlib import Path

# Define paths
DATA_PATH = Path("./data/wlasl-processed")
VIDEO_PATH = DATA_PATH / "videos"
ANNOTATION_FILE = Path("/home/adminjz/Project24-25/HandiSpeakV2/data/wlasl-processed/WLASL_v0.3.json")


# Function to load metadata
def load_metadata():
    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)
    # Print the structure to understand it
    print("Metadata structure:", type(data))
    if isinstance(data, list):
        print("Example entry:", data[0])  # Print the first entry if it's a list
    elif isinstance(data, dict):
        print("Top-level keys:", data.keys())
    return data

# Function to process videos
def process_videos(metadata):
    for video_entry in metadata["videos"]:
        video_file = VIDEO_PATH / video_entry["name"]
        if not video_file.exists():
            print(f"Video file {video_file} not found!")
            continue
        # Example: Extract and save frames
        extract_frames(video_file, output_dir=DATA_PATH / "frames")

# Function to extract frames
def extract_frames(video_file, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_file))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Save every 10th frame as an example
        if frame_count % 10 == 0:
            frame_path = output_dir / f"{video_file.stem}_frame_{frame_count}.jpg"
            cv2.imwrite(str(frame_path), frame)
        frame_count += 1
    cap.release()

# Main function
def main():
    print("Loading metadata...")
    metadata = load_metadata()
    print("Processing videos...")
    process_videos(metadata)
    print("Data preprocessing completed.")

if __name__ == "__main__":
    main()
