import os
import json
import random

# List of dataset directories to process
datasets = ["keypoints_aug_20", "keypoints_aug_50", "keypoints_aug_80"]
split_ratio = 0.8  # 80% train, 20% val

for dataset_dir in datasets:
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print(f"\n📁 Processing: {dataset_dir}")

    json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]

    for json_file in json_files:
        if os.path.exists(os.path.join(train_dir, json_file)) and os.path.exists(os.path.join(val_dir, json_file)):
            print(f"⏩ Skipping {json_file} (already split)")
            continue

        try:
            file_path = os.path.join(dataset_dir, json_file)
            with open(file_path, "r") as f:
                data = json.load(f)

            video_ids = list(data.keys())
            random.shuffle(video_ids)
            split_point = int(len(video_ids) * split_ratio)
            train_ids = video_ids[:split_point]
            val_ids = video_ids[split_point:]

            train_data = {vid: data[vid] for vid in train_ids}
            val_data = {vid: data[vid] for vid in val_ids}

            with open(os.path.join(train_dir, json_file), "w") as f:
                json.dump(train_data, f, indent=4)
            with open(os.path.join(val_dir, json_file), "w") as f:
                json.dump(val_data, f, indent=4)

            print(f"✅ {json_file}: Train {len(train_ids)}, Val {len(val_ids)}")

        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")

print("\n✅ All dataset splits complete!")
