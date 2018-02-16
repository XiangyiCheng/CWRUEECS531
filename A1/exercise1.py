import cv2
import numpy as np

img=cv2.imread('sydney.jpg',cv2.IMREAD_COLOR)
h,w=img.shape[:2]
#img[:h,:w]=[255,255,255]
cv2.imshow('original_image',img)
cv2.waitKey(0)
