# Correct Data Splitting Script
import os
import json
import random
import shutil

# Directories and split ratio
data_dir = 'keypoints_aug'
train_dir = 'keypoints_aug/train'
val_dir = 'keypoints_aug/val'
split_ratio = 0.8  # 80% train, 20% validation

# Create train and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Iterate through each JSON file (representing one word)
json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

for json_file in json_files:
    
    if os.path.exists(os.path.join(train_dir, json_file)) and os.path.exists(os.path.join(val_dir, json_file)):
        print(f"⏩ Skipping {json_file} (already split)")
        continue
    
    try:
        # Load the JSON file
        file_path = os.path.join(data_dir, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract all video IDs from the JSON file
        video_ids = list(data.keys())
        random.shuffle(video_ids)
        split_index = int(len(video_ids) * split_ratio)
        train_video_ids = video_ids[:split_index]
        val_video_ids = video_ids[split_index:]

        # Create separate dictionaries for train and val
        train_data = {vid: data[vid] for vid in train_video_ids}
        val_data = {vid: data[vid] for vid in val_video_ids}

        # Save the train JSON file
        train_file_path = os.path.join(train_dir, json_file)
        with open(train_file_path, 'w') as f:
            json.dump(train_data, f, indent=4)

        # Save the val JSON file
        val_file_path = os.path.join(val_dir, json_file)
        with open(val_file_path, 'w') as f:
            json.dump(val_data, f, indent=4)

        print(f"Processed {json_file}: Train videos {len(train_video_ids)}, Val videos {len(val_video_ids)}")

    except Exception as e:
        print(f"Error processing file {json_file}: {e}")

print("✅ Data splitting complete with balanced videos in both train and val!")