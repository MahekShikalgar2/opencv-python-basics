import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "Resources/cards.png"
img = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny Edge Detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area and get the largest one (assumed to be the card)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

if contours:
    largest_contour = contours[0]

    # Approximate contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # If it has four corners, it's likely a card
    if len(approx) == 4:
        for point in approx:
            cv2.circle(img, tuple(point[0]), 10, (0, 0, 255), -1)

# Convert BGR to RGB for displaying in matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the result
plt.figure(figsize=(8, 6))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Detected Card Corners")
plt.show()
