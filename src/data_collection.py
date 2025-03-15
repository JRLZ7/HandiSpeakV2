import kagglehub

# Locate the dataset without re-downloading
path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")

print("Dataset is located at:", path)