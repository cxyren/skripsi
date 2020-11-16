import cv2
import numpy as np

w=1920
h=1080
# Make empty black image
image=np.zeros((h,w,3),np.uint8)

image = cv2.circle(image, (10, 5), radius=5, color=(0, 0, 255), thickness=-1)

cv2.imwrite("test.jpg", image)