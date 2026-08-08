# Here is the simple coin counter. You may improve this program.
import cv2
import numpy as np

image = cv2.imread("coin_counting/coin dataset/coins (6).jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
gray = cv2.medianBlur(gray, 5)
blurred = cv2.GaussianBlur(gray, (11, 11), 2)

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,       
    minDist=30,   
    param1=66, # canny high threshold 
    param2=21, # accumulator threshold
    minRadius=23, 
    maxRadius=40
)

coin_count = 0
if circles is not None:
    circles = np.uint16(np.around(circles))
    coin_count = circles.shape[1]
    for (x, y, r) in circles[0, :]:
        # draw coin outline
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)

print("I count {} coins in this image".format(coin_count))

blur = blurred.copy()
coins = image.copy()
blur = cv2.resize(blurred, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
coins = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow("Blurred", blur)
cv2.imshow('Coins', coins)
cv2.waitKey(0)