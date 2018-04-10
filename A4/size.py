import numpy as np
import cv2

b=cv2.imread('block.png')
h,w,c=b.shape
print 'h=',h,'w=',w
resize=cv2.resize(b,(800,400))
cv2.imwrite('block_r.png',resize)
