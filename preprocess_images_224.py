import os
import cv2

# Original and output directories
DATA_DIR = "Kaggle_TB"
OUTPUT_DIR = "Kaggle_TB_resized"
NORMAL_DIR = os.path.join(DATA_DIR, "Normal")
TB_DIR = os.path.join(DATA_DIR, "Tuberculosis")

OUTPUT_NORMAL = os.path.join(OUTPUT_DIR, "Normal")
OUTPUT_TB = os.path.join(OUTPUT_DIR, "Tuberculosis")

IMG_SIZE = 224

# Create output directories if they don't exist
os.makedirs(OUTPUT_NORMAL, exist_ok=True)
os.makedirs(OUTPUT_TB, exist_ok=True)

def resize_and_save(input_dir, output_dir):
    for fname in os.listdir(input_dir):
        if fname.lower().endswith('.png'):
            img_path = os.path.join(input_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            save_path = os.path.join(output_dir, fname)
            cv2.imwrite(save_path, img_resized)
            print(f"Saved resized image: {save_path}")

# Resize and save Normal images
resize_and_save(NORMAL_DIR, OUTPUT_NORMAL)

# Resize and save TB images
resize_and_save(TB_DIR, OUTPUT_TB)

print("All images resized and saved successfully.")
