import albumentations as A
import cv2
import os
import glob
import random
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_DIR = "data/raw/distractors"       # Where you downloaded images in Step 2
OUTPUT_DIR = "data/synthetic_attacks"    # Where we save the "fake" duplicates
NUM_IMAGES_TO_ATTACK = 2000              # Don't process everything, just enough to test

# --- THE ATTACK PIPELINE ---
# This defines how we "damage" the image to simulate a repost.
# We apply ONE of these attacks to each image.
attack_pipeline = A.OneOf([
    # 1. Geometric Attacks (Hard for pixel matching)
    A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
    ], p=0.5),

    # 2. Quality/Noise Attacks (Hard for pixel matching)
    A.Compose([
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    ], p=0.5),

    # 3. Color/Filter Attacks (Simulates Instagram filters)
    A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.ToGray(p=0.3),
    ], p=0.5),
    
    # 4. Occlusion (Simulates watermarks/text overlays)
    A.CoarseDropout(
        max_holes=3, max_height=50, max_width=50, 
        min_holes=1, min_height=20, min_width=20, 
        fill_value=0, p=0.3
    ),
], p=1.0)  # p=1.0 means "Always apply one of these blocks"

def main():
    # 1. Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Find input images (Recursively search for.jpg and.jpeg)
    # This handles cases where the download might have created subfolders
    print(f"Scanning {INPUT_DIR} for images...")
    image_paths = glob.glob(os.path.join(INPUT_DIR, "**", "*.jpg"), recursive=True)
    image_paths += glob.glob(os.path.join(INPUT_DIR, "**", "*.jpeg"), recursive=True)
    
    if len(image_paths) == 0:
        print(f"ERROR: No images found in {INPUT_DIR}. Did Step 2 finish successfully?")
        return

    # 3. Shuffle and select a subset
    random.shuffle(image_paths)
    selected_paths = image_paths
    print(f"Generating synthetic duplicates for {len(selected_paths)} images...")

    count = 0
    for img_path in tqdm(selected_paths):
        try:
            # Read image using OpenCV
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Convert BGR (OpenCV format) to RGB (Albumentations format)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply the attack
            augmented = attack_pipeline(image=image)['image']
            
            # Convert back to BGR to save it
            augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            
            # Save the file
            # Naming convention: original_filename_attack.jpg
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            save_name = f"{name}_attack.jpg"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            
            cv2.imwrite(save_path, augmented)
            count += 1
            
        except Exception as e:
            # Skip images that are too small or corrupt
            pass

    print(f"\nSUCCESS: Generated {count} synthetic near-duplicates.")
    print(f"You can find them in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()