import cv2
import numpy as np

img_shapes=cv2.imread('shapes.jpg')
# convert the image into black and white.
img_shapes_gray=cv2.cvtColor(img_shapes,cv2.COLOR_BGR2GRAY)
template=cv2.imread('template.jpg',0)
w,h=template.shape[::-1]

# detect the object from the template. 
result=cv2.matchTemplate(img_shapes_gray,template,cv2.TM_CCOEFF_NORMED)
# set the threshold.
threshold=0.2
# set the detection numbers
count=0

# get the coordinates of the targets in the image.
location=np.where(result>=threshold)

# draw a rectangle in the detected location.
for pt in zip(*location[::-1]):
	cv2.rectangle(img_shapes,pt,(pt[0]+w,pt[1]+h),(255,255,0),1)
	count+=1

print 'detection numbers=',count

cv2.imshow('shapes',img_shapes)
cv2.waitKey()
