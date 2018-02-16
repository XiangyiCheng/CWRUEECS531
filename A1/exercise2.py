import cv2
import numpy as np

img=cv2.imread('sydney.jpg')
cv2.imshow('img',img)
canny200=cv2.Canny(img,200,200)
cv2.imshow('img1',canny200)
canny300=cv2.Canny(img,300,300)
cv2.imshow('img2',canny300)
canny500=cv2.Canny(img,500,500)
cv2.imshow('img3',canny500)

laplacian=cv2.Laplacian(img,cv2.CV_64F)
cv2.imshow('img4',laplacian)

sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
cv2.imshow('img5',sobelx)
cv2.imshow('img6',sobely)

cv2.waitKey()
