import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)  # Use 0 if only one camera is connected
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass


cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)

# Corrected trackbar names
cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)  # Fixed name
cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)  # Fixed name
cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)  # Fixed name

while True:
    ret, img = cap.read()  # Unpacking properly
    if not ret:  # Check if the frame was captured successfully
        print("Failed to grab frame. Exiting...")
        break

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Corrected trackbar names
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    hstack = np.hstack([img,mask,result])

    # cv2.imshow("Original", img)
    # cv2.imshow("HSV color", imgHSV)
    # cv2.imshow("Result", result)
    cv2.imshow("Horizantal Stacking", hstack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
