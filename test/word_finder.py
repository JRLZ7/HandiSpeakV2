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

# Get top 50, 100, 150, 200 words
top_50 = sorted_words[:50]
top_100 = sorted_words[:100]
top_150 = sorted_words[:150]
top_200 = sorted_words[:200]

# Print the results

print("\nTop 50 words:")
for word, count in top_50:
    print(f"{word}: {count}")

print("\nTop 100 words:")
for word, count in top_100:
    print(f"{word}: {count}")

print("\nTop 150 words:")
for word, count in top_150:
    print(f"{word}: {count}")

print("\nTop 200 words:")
for word, count in top_200:
    print(f"{word}: {count}")

# Save the top words to JSON files
os.makedirs(os.path.expanduser("~/Projects/HandiSpeakV2/datasets/"), exist_ok=True)
with open(os.path.expanduser("~/Projects/HandiSpeakV2/datasets/top_50_words.json"), "w") as f:
    json.dump([word for word, _ in top_50], f, indent=4)
with open(os.path.expanduser("~/Projects/HandiSpeakV2/datasets/top_100_words.json"), "w") as f:
    json.dump([word for word, _ in top_100], f, indent=4)
with open(os.path.expanduser("~/Projects/HandiSpeakV2/datasets/top_150_words.json"), "w") as f:
    json.dump([word for word, _ in top_150], f, indent=4)
with open(os.path.expanduser("~/Projects/HandiSpeakV2/datasets/top_200_words.json"), "w") as f:
    json.dump([word for word, _ in top_200], f, indent=4)

print("\n✅ Top words saved to JSON files.")
