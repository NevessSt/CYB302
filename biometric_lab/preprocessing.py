import cv2
import matplotlib.pyplot as plt

image_path = "dataset/class img/1_1.jpg"

img = cv2.imread(image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

equalized = cv2.equalizeHist(gray)

cv2.imwrite("screenshots/grayscale.jpg", gray)
cv2.imwrite("screenshots/equalized.jpg", equalized)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")

plt.subplot(1,2,2)
plt.imshow(equalized, cmap="gray")
plt.title("Histogram Equalized")

plt.show()