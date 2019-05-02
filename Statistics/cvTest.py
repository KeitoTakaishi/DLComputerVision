import cv2
import numpy as np

img = np.random.normal(0, 1, (480, 640))*255.0
#img = cv2.imread('1000.png')
#img = cv2.imread('lena50.jpg')
print(img.shape)
print(img.dtype)
cv2.imwrite('test.png', img)
