from matplotlib import pyplot as plt
import numpy as np
import cv2

image = cv2.imread("image.jpg")
cropped = image[0:900, 0:600]
cv2.imshow("Cropped Image", cropped)

chans = cv2.split(cropped)
colors = ("b", "g", "r")
plt.figure()
plt.title("Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color = color)
    plt.xlim([0, 256])

min_price, max_price = 5, 20
color_pixels = np.sum(hist[11:245])
ratio = color_pixels / np.sum(hist)
price = min_price + ratio * (max_price - min_price)
print(f"Printing Price: PHP {price:.2f}")

plt.show()
cv2.waitKey(0)