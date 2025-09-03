import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Dataset folder (update if different)
DATA_DIR = "./Kaggle_TB_resized"  # Relative path assuming dataset is in the same folder

NORMAL_DIR = os.path.join(DATA_DIR, "Normal")
TB_DIR = os.path.join(DATA_DIR, "Tuberculosis")

image_paths = []
labels = []

print("Listing Normal images...")
for fname in os.listdir(NORMAL_DIR):
    if fname.lower().endswith('.png'):
        image_paths.append(os.path.join(NORMAL_DIR, fname))
        labels.append(0)

print("Listing Tuberculosis images...")
for fname in os.listdir(TB_DIR):
    if fname.lower().endswith('.png'):
        image_paths.append(os.path.join(TB_DIR, fname))
        labels.append(1)

print(f"Total images: {len(image_paths)}")

images = []
for idx, path in enumerate(image_paths):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    images.append(img)
    if (idx + 1) % 500 == 0:
        print(f"Processed {idx + 1} images")

X = np.array(images)[..., np.newaxis]
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Preprocessing complete!")
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Save preprocessed data if desired (optional)
# np.savez_compressed("preprocessed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)

# If you want to save labels as well
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Preprocessed data and labels saved as .npy files.")