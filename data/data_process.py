
import glob
import cv2
import numpy as np

# Define the paths
data_path = "/data/raw/*.jpg"
save_path = "/data/processed/new_data2000.npy"
max_images = 2000

X_train = []

file_paths = glob.glob(data_path)

for i, file in enumerate(file_paths):
    if i >= max_images:
        break
    img = cv2.imread(file)
    if img is not None:
        X_train.append(img)

# Convert to np array and normalize
X_train = np.array(X_train)
X_train = X_train / 255.0

# The preprocessed data to use in training model
np.save(save_path, X_train)

print("Shape of X_train:", X_train.shape)
