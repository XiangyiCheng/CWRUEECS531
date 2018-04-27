import cv2
import numpy as np

img1=cv2.imread('/home/pi/Intubot/cascade_training/test/2.jpg')
img2= cv2.imread('/home/pi/Intubot/interface/intubot.JPG')


height,width=img2.shape[:2]
new_height=500.0
new_width=width/(height/new_height)

img2=cv2.resize(img2,(int(new_width),int(new_height)))
rows,cols,channels=img1.shape
roi=img1[0:rows,0:cols]
img2[0:rows,0:cols]=roi

cv2.imwrite('/home/pi/Intubot/cascade_training/create_test_img.jpg',img2)
cv2.imshow('img2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
