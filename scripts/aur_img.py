import cv2
import os
import numpy as np
import imgaug.augmenters as iaa
import random

# Input and output directories
input_folder = "../dataset/images"
output_folder = "../dataset/augmented_images"

# Ensure output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define augmentation techniques
augmenters = [
    iaa.GaussianBlur(sigma=(0, 1.5)),  # Add slight blur
    iaa.Affine(rotate=(-15, 15)),  # Rotate images randomly
    iaa.Fliplr(0.5),  # Flip horizontally with 50% chance
    iaa.AdditiveGaussianNoise(scale=(5, 20)),  # Add noise
    iaa.Multiply((0.7, 1.3)),  # Random brightness
    iaa.Crop(percent=(0, 0.1))  # Random cropping
]

# Load images and apply augmentations
image_files = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

if not image_files:
    print("⚠️ No images found in dataset/images/ folder! Check if the folder is empty.")

for idx, image_file in enumerate(image_files):
    img_path = os.path.join(input_folder, image_file)
    image = cv2.imread(img_path)

    if image is None:
        print(f"⚠️ Skipping {image_file} (could not load)")
        continue

    print(f"🔄 Processing {image_file}...")

    # Apply random augmentations
    for i in range(5):  # Generate 5 variations per image
        aug_image = random.choice(augmenters)(image=image)
        aug_img_name = f"aug_{idx}_{i}.jpg"
        aug_img_path = os.path.join(output_folder, aug_img_name)

        cv2.imwrite(aug_img_path, aug_image)
        print(f"✅ Saved: {aug_img_path}")

print("🚀 Augmentation Completed!")
