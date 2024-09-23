import os
import cv2
import numpy as np

def clean_data(data_dir):
    """
    Cleans the collected data by removing images that are mostly black (no hand landmarks).
    """
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            if np.mean(image) < 10:  # Arbitrary threshold, adjust as necessary
                print(f'Removing {image_path} due to lack of content.')
                os.remove(image_path)

if __name__ == "__main__":
    DATA_DIR = './data/ASL'
    clean_data(DATA_DIR)