import cv2
import numpy as np
import os

# Path to image folder and mask
image_folder = 'Adaptive_threshold'      # Folder containing images
mask_path = '/home/sneha/Selent/final_sealant/final_mask_401.jpg'        # Path to the single binary mask
output_folder = 'mask_401'     # Folder to save output

# Load the mask
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise FileNotFoundError(f"Mask not found at: {mask_path}")

# Ensure binary mask
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all image files in the image folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in image_files:
    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping {filename}: cannot read image.")
        continue

    # Resize mask to match image size if needed
    if (mask.shape[0] != image.shape[0]) or (mask.shape[1] != image.shape[1]):
        resized_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        resized_mask = binary_mask

    # Convert mask to 3 channels if image is color
    if len(image.shape) == 3 and image.shape[2] == 3:
        mask_3ch = cv2.merge([resized_mask, resized_mask, resized_mask])
    else:
        mask_3ch = resized_mask

    # Apply mask
    masked_image = cv2.bitwise_and(image, mask_3ch)

    # # Save result
    # output_path = os.path.join(output_folder, filename)
    # cv2.imwrite(output_path, masked_image)
    # print(f"Saved masked image: {output_path}")

        # Save result as .png
    output_filename = os.path.splitext(filename)[0] + '.png'  # Change extension to .png
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, masked_image)
    print(f"Saved masked image: {output_path}")

