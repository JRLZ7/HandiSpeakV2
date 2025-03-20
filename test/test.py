import json
import os

metadata_path = "~/Projects/HandiSpeakV2/data/WLASL/WLASL_v0.3.json"

with open(os.path.expanduser(metadata_path), "r") as file:
    data = json.load(file)

# Print a few sample video IDs and URLs
for word in data[:5]:  # Checking first 5 words
    print(f"Word: {word['gloss']}")
    for instance in word['instances'][:3]:  # Checking first 3 instances
        print(f"  Video ID: {instance['video_id']}, URL: {instance['url']}")
