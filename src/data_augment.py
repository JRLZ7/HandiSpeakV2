import os
import json
import random
import copy
from tqdm import tqdm

def augment_keypoints(keypoints, noise_level=0.01):
    augmented = copy.deepcopy(keypoints)
    for frame in augmented:
        for part in frame.keys():
            if isinstance(frame[part], list):
                for point in frame[part]:
                    if 'x' in point and 'y' in point and 'z' in point:
                        point['x'] += random.uniform(-noise_level, noise_level)
                        point['y'] += random.uniform(-noise_level, noise_level)
                        point['z'] += random.uniform(-noise_level, noise_level)
    return augmented

def augment_videos(word, input_dir, output_dir, target_count=50):
    try:
        input_path = os.path.join(input_dir, f"{word}.json")
        output_path = os.path.join(output_dir, f"{word}.json")

        # ✅ Skip if already augmented
        if os.path.exists(output_path):
            print(f"⏩ Skipping '{word}' (already augmented)")
            return

        if not os.path.exists(input_path):
            print(f"❌ JSON file not found: {input_path}")
            return
        
        with open(input_path, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, dict):
                    print(f"🚨 Invalid JSON format for word '{word}': not a dictionary.")
                    return
            except json.JSONDecodeError as e:
                print(f"🚨 JSON decode error for word '{word}': {e}")
                return

        if isinstance(data, dict):
            video_ids = list(data.keys())
        else:
            print(f"🚨 Unexpected structure: '{word}.json' is not a dictionary at top level.")
            return
        num_videos = len(video_ids)

        
        if num_videos == 0:
            print(f"🚨 No videos found for word '{word}'")
            return

        print(f"[INFO] Augmenting '{word}' (has {num_videos} videos), aiming for {target_count} total.")

        augmented_data = copy.deepcopy(data)
        aug_count = target_count - num_videos

        for i in range(aug_count):
            original_video_id = random.choice(video_ids)
            augmented_video_id = f"aug_{i}_{original_video_id}"
            
            # Check for valid video data structure
            original_keypoints = data.get(original_video_id)
            if not isinstance(original_keypoints, list):
                print(f"🚨 Invalid keypoints structure for video '{original_video_id}' in word '{word}'")
                continue

            augmented_keypoints = augment_keypoints(original_keypoints)
            augmented_data[augmented_video_id] = augmented_keypoints
            
            print(f"  [INFO] Augmentation progress for '{word}': {i+1}/{aug_count} videos")

        with open(output_path, "w") as out_file:
            json.dump(augmented_data, out_file)
        
        print(f"[INFO] Saved augmented data for '{word}' to {output_path}")

    except Exception as e:
        print(f"🚨 Error during augmentation for '{word}': {e}")

def main(input_dir="keypoints", output_dir="keypoints_aug", target_count=50):
    os.makedirs(output_dir, exist_ok=True)
    words = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith(".json")]

    for word in tqdm(words, desc="Processing words"):
        augment_videos(word, input_dir, output_dir, target_count)

    print("✅ Augmentation complete for all words!")

if __name__ == "__main__":
    main()
