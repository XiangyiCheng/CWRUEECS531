from scipy import misc
import numpy as np
import cv2

img=misc.imread('shapes.jpg',mode='L')
noisy1=img+3*img.std()*np.random.random(img.shape)
alot=5*img.max()*np.random.random(img.shape)
noisy2=img+alot

img1=cv2.imwrite('noise1.jpg',noisy1)
img2=cv2.imwrite('noise3.jpg',noisy2)

