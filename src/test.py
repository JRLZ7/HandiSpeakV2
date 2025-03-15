import shutil
import os

# Define paths
source_path = "/home/adminjz/Project24-25/HandiSpeakV2/data/MS-ASL/MS-ASL"
destination_path = "/home/adminjz/Project24-25/HandiSpeakV2/data/MS-ASL"

# Move all files from the extra MS-ASL directory to the correct location
for filename in os.listdir(source_path):
    shutil.move(os.path.join(source_path, filename), destination_path)

# Remove the now-empty MS-ASL directory
os.rmdir(source_path)

print("✅ Extra 'MS-ASL' folder removed successfully!")
