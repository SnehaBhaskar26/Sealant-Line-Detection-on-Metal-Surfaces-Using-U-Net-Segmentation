# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the image
# image_path = "/home/sneha/Selent/dataset/401/9724_-_401_cam1_20.jpg"  # Update the path if needed
# image = cv2.imread(image_path)
 
# # Convert to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #img = cv2.medianBlur(gray, 3)
# img = cv2.GaussianBlur(gray, (5,5), 0)

# # Apply Adaptive Thresholding to highlight the black sealant
# thresh = cv2.adaptiveThreshold(
# img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
# )
# cv2.imwrite("thre3.jpg",thresh)




import cv2
import os

def process_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # List all image files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Skipping unreadable image: {filename}")
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            img = cv2.GaussianBlur(gray, (5, 5), 0)
            #img = cv2.medianBlur(gray, 3)

            # Apply Adaptive Thresholding
            thresh = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )

            # Save the output image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, thresh)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_folder = "/home/sneha/Selent/final_sealant/401"  # Update this path
    output_folder = "/home/sneha/Selent/final_sealant/Adaptive_threshold"  # Update or create this folder

    process_images_in_folder(input_folder, output_folder)
