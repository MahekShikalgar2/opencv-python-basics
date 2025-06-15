import cv2
from numpy.ma.core import shape

path = "Resources/lena.png"
img = cv2.imread(path)
print(img.shape)

width ,height = 1000, 1000
imgResize = cv2.resize(img,(width,height))
print(imgResize.shape)

imgCropped = img[300:900,430:470]
imhCropResize = cv2.resize(imgCropped,(img.shape[1],img.shape[0]))

cv2.imshow("img",img)
cv2.imshow("img Resized",imgResize)
cv2.imshow("img Cropped",imgCropped)
cv2.imshow("img Cropped Resize",imhCropResize)
cv2.waitKey(0)