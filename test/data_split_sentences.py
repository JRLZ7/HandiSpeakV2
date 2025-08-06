import json
import random
from pathlib import Path

input_file = "synthetic_sentences_50/synthetic_50_sentences.json"
output_dir = Path("synthetic_sentences_50")

with open(input_file, "r") as f:
    data = json.load(f)

random.shuffle(data)
split_idx = int(0.8 * len(data))

train_data = data[:split_idx]
val_data = data[split_idx:]

with open(output_dir / "synthetic_50_train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open(output_dir / "synthetic_50_val.json", "w") as f:
    json.dump(val_data, f, indent=2)

print("✅ Saved 400 train / 100 val samples.")
