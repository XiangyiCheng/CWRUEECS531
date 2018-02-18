import cv2
import numpy as np

img=cv2.imread('sydney.jpg',cv2.IMREAD_COLOR)
img_blurred=img.copy()
#img_color=cv2.imread('sydney.jpg',cv2.IMREAD_COLOR)
h,w=img.shape[:2]

Gaussian=1/256.*np.matrix('1;4;6;4;1')*np.matrix('1 4 6 4 1')
center_poc=(len(Gaussian)-1)/2
img_reshape=np.zeros((h+2*center_poc,w+2*center_poc,3))
for k in range(0,h):
	for l in range(0,w):
		for d in range(0,3):
			img_reshape[k+center_poc,l+center_poc,d]=img_blurred.item(k,l,d)
	

for c in range(0,3):
	for i in range(0,h):
		for j in range(0,w):
			sum=0
			for h1 in range(0,len(Gaussian)):
				for w1 in range(0,len(Gaussian)):
					pixel_bg=img_reshape.item(i+h1,j+w1,c)
					pixel_kernel=Gaussian[h1,w1]
					sum=sum+(pixel_bg*pixel_kernel) 
			img_blurred.itemset((i,j,c),sum)

cv2.imwrite('sydney_blurred_colored.jpg',img_blurred)

cv2.imshow('gray_image',img)
cv2.imshow('blurred_image',img_blurred)
cv2.waitKey(0)
