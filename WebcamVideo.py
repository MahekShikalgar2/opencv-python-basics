import cv2
import numpy as np


# Function to stack images
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)

    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)

                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)

        hor = [np.hstack(imgArray[row]) for row in range(rows)]
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)

            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)

        ver = np.hstack(imgArray)

    return ver


# Open webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Set resolution (optional)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Kernel for dilation and erosion
kernel = np.ones((5, 5), np.uint8)

while True:
    success, img = cap.read()  # Read frame from webcam
    if not success:
        print("Error: Could not access webcam")
        break

    # Image Processing
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)
    imgCanny = cv2.Canny(imgBlur, 100, 150)
    imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
    imgEroded = cv2.erode(imgDilation, kernel, iterations=1)

    # Stack images together
    stackedImages = stackImages(0.6, [[img, imgGray, imgBlur], [imgCanny, imgDilation, imgEroded]])

    # Show the stacked images
    cv2.imshow("Webcam - Processed Images", stackedImages)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
