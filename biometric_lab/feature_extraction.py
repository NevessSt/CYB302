import cv2
import os
import numpy as np

# Dataset path
dataset_path = "dataset/archive/ORL-Organised"

faces = []
labels = []

print("Loading images...")

# Read every person's folder
for person_folder in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person_folder)

    if os.path.isdir(person_path):

        label = int(person_folder)

        for image_name in os.listdir(person_path):

            image_path = os.path.join(person_path, image_name)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:

                faces.append(image)
                labels.append(label)

print(f"Loaded {len(faces)} images.")
print(f"Loaded {len(set(labels))} people.")

# Convert labels into NumPy array
labels = np.array(labels)

# Create the LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Training model...")

recognizer.train(faces, labels)

# Save the trained model
recognizer.save("biometric_lab/trainer.yml")

print("Training Complete!")
print("Model saved as trainer.yml")