import cv2
import numpy as np

# Read images
img1 = cv2.imread("Resources/anim.png")
img2 = cv2.imread("Resources/Muna.png")

# Check if images are loaded properly
if img1 is None or img2 is None:
    print("Error: One or both images not found!")
    exit()

print("Original shape of img1:", img1.shape)
print("Original shape of img2:", img2.shape)

# Resize both images to the same size
width, height = 300, 300  # Set a common width and height
img1 = cv2.resize(img1, (width, height))
img2 = cv2.resize(img2, (width, height))

# Stack images horizontally and vertically
hor = np.hstack((img1, img2))
ver = np.vstack((img1, img2))

# Display images
cv2.imshow("Horizontal", hor)
cv2.imshow("Vertical", ver)

cv2.waitKey(0)
cv2.destroyAllWindows()
