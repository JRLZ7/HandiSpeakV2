import json
import os

# Path to metadata file
metadata_path = "~/Projects/HandiSpeakV2/data/WLASL/WLASL_v0.3.json"
missing_videos_path = "~/Projects/HandiSpeakV2/data/WLASL/missing.txt"

# Load the metadata
with open(os.path.expanduser(metadata_path), "r") as file:
    data = json.load(file)

# Load missing videos
missing_videos = set()
if os.path.exists(os.path.expanduser(missing_videos_path)):
    with open(os.path.expanduser(missing_videos_path), "r") as file:
        missing_videos = set(line.strip() for line in file if line.strip())

print(f"Loaded {len(missing_videos)} missing videos from {missing_videos_path}")

# Dictionary to store word counts
word_counts = {}

# Iterate through each word in the dataset
for word in data:
    label = word['gloss']
    videos = word['instances']

    # Count valid videos (excluding missing ones)
    valid_videos = [
        video for video in videos
        if video['video_id'] not in missing_videos and "youtube" not in video['url']
    ]
    
    count = len(valid_videos)
    if count > 0:
        word_counts[label] = count

# Sort words by the number of videos in descending order
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# Get top 20, 50, and 80 words
top_20 = sorted_words[:20]
top_50 = sorted_words[:50]
top_80 = sorted_words[:80]

# Print the results
print("\nTop 20 words:")
for word, count in top_20:
    print(f"{word}: {count}")

print("\nTop 50 words:")
for word, count in top_50:
    print(f"{word}: {count}")

print("\nTop 80 words:")
for word, count in top_80:
    print(f"{word}: {count}")

# Save the top words to JSON files
os.makedirs(os.path.expanduser("~/Projects/HandiSpeakV2/datasets/"), exist_ok=True)
with open(os.path.expanduser("~/Projects/HandiSpeakV2/datasets/top_20_words.json"), "w") as f:
    json.dump([word for word, _ in top_20], f, indent=4)
with open(os.path.expanduser("~/Projects/HandiSpeakV2/datasets/top_50_words.json"), "w") as f:
    json.dump([word for word, _ in top_50], f, indent=4)
with open(os.path.expanduser("~/Projects/HandiSpeakV2/datasets/top_80_words.json"), "w") as f:
    json.dump([word for word, _ in top_80], f, indent=4)

print("\n✅ Top words saved to JSON files.")
