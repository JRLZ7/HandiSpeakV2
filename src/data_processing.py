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

# Function to extract metadata for N words
def extract_words_metadata(num_words):
    words = sorted(data, key=lambda x: len(x['instances']), reverse=True)[:num_words]
    extracted_data = {}

    for word in words:
        label = word['gloss']
        videos = word['instances']
        extracted_data[label] = []

        for video in videos:
            video_id = video['video_id']

            # Skip if video ID is in missing.txt
            if video_id in missing_videos:
                print(f"❌ Skipping missing video ID: {video_id}")
                continue

            url = video['url']
            if "youtube" not in url:  # Exclude YouTube links
                # Print to see what the video ID looks like
                print(f"Label: {label}, Video ID: {video_id}")

                # Try different possible file name formats
                file_names = [
                    f"{video_id}.mp4",
                ]

                # Check each possible file name
                for file_name in file_names:
                    file_path = os.path.realpath(os.path.expanduser(f"~/Projects/HandiSpeakV2/data/WLASL/videos/{file_name}"))
                    print(f"Checking: {file_path}")  # Debug print
                    if os.path.exists(file_path):
                        print(f"✅ Found file: {file_path}")  # Debug print
                        extracted_data[label].append(file_name)
                        break  # Stop checking more formats if found
                    else:
                        print(f"❌ File not found: {file_path}")

    return extracted_data

# Generate datasets for 20, 50, 100, and 200 words
datasets = {
    "20_words": extract_words_metadata(20),
    "50_words": extract_words_metadata(50),
    "100_words": extract_words_metadata(100),
    "200_words": extract_words_metadata(200),
}

# Save extracted metadata
for name, dataset in datasets.items():
    output_path = f"~/Projects/HandiSpeakV2/datasets/{name}_metadata.json"
    os.makedirs(os.path.dirname(os.path.expanduser(output_path)), exist_ok=True)  # Fix: Create parent directory
    with open(os.path.expanduser(output_path), "w") as outfile:
        json.dump(dataset, outfile, indent=4)

print("Metadata for 20, 50, 100, and 200 words extracted and saved.")


# import json
# import os

# # Path to metadata file
# metadata_path = "~/Projects/HandiSpeakV2/data/WLASL/WLASL_v0.3.json"

# # Load the metadata
# with open(os.path.expanduser(metadata_path), "r") as file:
#     data = json.load(file)

# # Function to extract metadata for N words
# def extract_words_metadata(num_words):
#     words = sorted(data, key=lambda x: len(x['instances']), reverse=True)[:num_words]
#     extracted_data = {}

#     for word in words:
#         label = word['gloss']
#         videos = word['instances']
#         extracted_data[label] = []

#         for video in videos:
#             video_id = video['video_id']
#             url = video['url']
#             if "youtube" not in url:  # Exclude YouTube links

#                 # Print to see what the video ID looks like
#                 print(f"Label: {label}, Video ID: {video_id}")

#                 # Try different possible file name formats
#                 file_names = [
#                     f"{video_id}.mp4",
#                 ]

#                 # Check each possible file name
#                 for file_name in file_names:
#                     file_path = os.path.realpath(os.path.expanduser(f"~/Projects/HandiSpeakV2/data/WLASL/videos/{file_name}"))
#                     print(f"Checking: {file_path}")  # Debug print
#                     if os.path.exists(file_path):
#                         print(f"✅ Found file: {file_path}")  # Debug print
#                         extracted_data[label].append(file_name)
#                         break  # Stop checking more formats if found
#                     else:
#                         print(f"❌ File not found: {file_path}")

#     return extracted_data




# # Generate datasets for 20, 50, 100, and 200 words
# datasets = {
#     "20_words": extract_words_metadata(20),
#     "50_words": extract_words_metadata(50),
#     "100_words": extract_words_metadata(100),
#     "200_words": extract_words_metadata(200),
# }

# # Save extracted metadata
# for name, dataset in datasets.items():
#     output_path = f"~/Projects/HandiSpeakV2/datasets/{name}_metadata.json"
#     os.makedirs(os.path.dirname(os.path.expanduser(output_path)), exist_ok=True)  # Fix: Create parent directory
#     with open(os.path.expanduser(output_path), "w") as outfile:
#         json.dump(dataset, outfile, indent=4)

# print("Metadata for 20, 50, 100, and 200 words extracted and saved.")
