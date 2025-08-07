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

# Load word lists from individual JSON files
word_files = {
    "top_50": "~/Projects/HandiSpeakV2/datasets/top_50_words.json",
    "top_100": "~/Projects/HandiSpeakV2/datasets/top_100_words.json",
    "top_150": "~/Projects/HandiSpeakV2/datasets/top_150_words.json",
    "top_200": "~/Projects/HandiSpeakV2/datasets/top_200_words.json",
}

# Function to load words from a JSON file
def load_words(file_path):
    with open(os.path.expanduser(file_path), "r") as file:
        return json.load(file)

# Function to extract metadata for a given word list
def extract_words_metadata(word_list):
    extracted_data = {}

    for word in word_list:
        word_data = next((item for item in data if item['gloss'] == word), None)
        if not word_data:
            print(f"❌ Word not found in metadata: {word}")
            continue

        label = word_data['gloss']
        videos = word_data['instances']
        extracted_data[label] = []

        for video in videos:
            video_id = video['video_id']

            # Skip if video ID is in missing.txt
            if video_id in missing_videos:
                print(f"❌ Skipping missing video ID: {video_id}")
                continue

            url = video['url']
            if "youtube" not in url:  # Exclude YouTube links
                print(f"Label: {label}, Video ID: {video_id}")

                file_name = f"{video_id}.mp4"
                file_path = os.path.realpath(os.path.expanduser(f"~/Projects/HandiSpeakV2/data/WLASL/videos/{file_name}"))
                if os.path.exists(file_path):
                    print(f"✅ Found file: {file_path}")
                    extracted_data[label].append(file_name)
                else:
                    print(f"❌ File not found: {file_path}")

    return extracted_data

# Generate datasets for "N" words
for name, file_path in word_files.items():
    words = load_words(file_path)
    dataset = extract_words_metadata(words)

    output_path = f"~/Projects/HandiSpeakV2/datasets/{name}_metadata.json"
    os.makedirs(os.path.dirname(os.path.expanduser(output_path)), exist_ok=True)
    with open(os.path.expanduser(output_path), "w") as outfile:
        json.dump(dataset, outfile, indent=4)

    print(f"✅ Metadata for {name} words extracted and saved.")
