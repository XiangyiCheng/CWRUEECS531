import numpy as np
import cv2

b=cv2.imread('tri.JPG')
h,w,c=b.shape
print 'h=',h,'w=',w
resize=cv2.resize(b,(600,418))
cv2.imwrite('tri2.png',resize)
