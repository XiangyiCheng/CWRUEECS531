import numpy as np
import cv2

b=cv2.imread('../Final_Project/haar.png')
h,w,c=b.shape
print 'h=',h,'w=',w
resize=cv2.resize(b,(400,250))
cv2.imwrite('../Final_Project/haar2.png',resize)
