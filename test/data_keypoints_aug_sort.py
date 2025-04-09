import os
import shutil
import json
from pathlib import Path

# Mapping: metadata file -> destination folder
word_sets = {
    "/home/jason/Projects/HandiSpeakV2/datasets/top_20_words.json": "keypoints_aug_20",
    "/home/jason/Projects/HandiSpeakV2/datasets/top_50_words.json": "keypoints_aug_50",
    "/home/jason/Projects/HandiSpeakV2/datasets/top_80_words.json": "keypoints_aug_80",
}

# Source directory
src_dir = Path("/home/jason/Projects/HandiSpeakV2/keypoints_aug")

for word_file, dest_folder in word_sets.items():
    # Load list of words
    with open(word_file, "r") as f:
        words = json.load(f)

    # Create destination folder
    dest_path = Path(dest_folder)
    dest_path.mkdir(parents=True, exist_ok=True)

    for word in words:
        src_file = src_dir / f"{word}.json"
        dst_file = dest_path / f"{word}.json"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            print(f"✅ Copied {word}.json → {dest_folder}/")
        else:
            print(f"⚠️ Missing: {src_file}")

print("✅ Done organizing word JSONs into 20/50/80 folders.")
