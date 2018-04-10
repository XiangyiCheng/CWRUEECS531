import numpy as np
import cv2

b=cv2.imread('eq3.jpg')
h,w,c=b.shape
print 'h=',h,'w=',w
resize=cv2.resize(b,(250,90))
cv2.imwrite('eq3_r4.jpg',resize)
