import cv2
import numpy as np

img_shapes=cv2.imread('shapes.jpg')
img_shapes_gray=cv2.cvtColor(img_shapes,cv2.COLOR_BGR2GRAY)
template=cv2.imread('template.jpg',0)
w,h=template.shape[::-1]

result=cv2.matchTemplate(img_shapes_gray,template,cv2.TM_CCOEFF_NORMED)
threshold=0.2
location=np.where(result>=threshold)

for pt in zip(*location[::-1]):
	cv2.rectangle(img_shapes,pt,(pt[0]+w,pt[1]+h),(255,255,0),1)

cv2.imshow('shapes',img_shapes)
cv2.waitKey()
