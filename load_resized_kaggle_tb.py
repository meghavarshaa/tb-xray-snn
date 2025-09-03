import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Updated data directories
DATA_DIR = "Kaggle_TB_resized"
NORMAL_DIR = os.path.join(DATA_DIR, "Normal")
TB_DIR = os.path.join(DATA_DIR, "Tuberculosis")

# Collect image paths and labels (no resizing needed)
image_paths = []
labels = []

for fname in os.listdir(NORMAL_DIR):
    if fname.lower().endswith('.png'):
        image_paths.append(os.path.join(NORMAL_DIR, fname))
        labels.append(0)  # Normal label

for fname in os.listdir(TB_DIR):
    if fname.lower().endswith('.png'):
        image_paths.append(os.path.join(TB_DIR, fname))
        labels.append(1)  # TB label

# Load and normalize images (no resize)
images = []
for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0  # Normalize pixel values to [0,1]
    images.append(img)

X = np.array(images)[..., np.newaxis]  # Shape: (N, 224, 224, 1)
y = np.array(labels)

# Split dataset into train/test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Data loaded from resized folders!")
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
