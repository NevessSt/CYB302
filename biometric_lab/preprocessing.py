import cv2
import os
import matplotlib.pyplot as plt
# Folder containing the images (adjust this for each person’s folder)
image_folder = "dataset/archive/ORL-Organised/1"  # Change '1' to each subject folder as needed
# Create a folder for screenshots if it doesn't exist
os.makedirs("screenshots", exist_ok=True)
# Loop through all files in the folder
for img_name in os.listdir(image_folder):
    if img_name.endswith(".jpg") or img_name.endswith(".png"):
        image_path = os.path.join(image_folder, img_name)
       
        # Read the image
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 100))
        gaussian = cv2.GaussianBlur(resized, (5, 5), 0)
        equalized = cv2.equalizeHist(gaussian)

        # Save grayscale image
        gray_output_name = f"screenshots/{os.path.splitext(img_name)[0]}_grayscale.jpg"
        cv2.imwrite(gray_output_name, gray)
        # Save resized image
        resized_output_name = f"screenshots/{os.path.splitext(img_name)[0]}_resized.jpg"
        cv2.imwrite(resized_output_name, resized)
        # Save Gaussian blur image
        gaussian_output_name = f"screenshots/{os.path.splitext(img_name)[0]}_gaussian.jpg"
        cv2.imwrite(gaussian_output_name, gaussian)
        # Save equalized image
        equalized_output_name = f"screenshots/{os.path.splitext(img_name)[0]}_equalized.jpg"
        cv2.imwrite(equalized_output_name, equalized)
print("Processing complete. Press Enter to exit...")
input()