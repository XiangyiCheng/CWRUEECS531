import cv2
import numpy as np

img=cv2.imread('sydney.jpg',cv2.IMREAD_GRAYSCALE)
img_blurred=img.copy()
h,w=img.shape[:2]

Gaussian=1/256.*np.matrix('1;4;6;4;1')*np.matrix('1 4 6 4 1')
center_poc=(len(Gaussian)-1)/2

# add two rows above and below the image respectively. add two colums on left and right sides of the image.
# reshape a zero matrix in (h+2,w+2) dimension.
img_reshape=np.zeros((h+2*center_poc,w+2*center_poc))

# give the original pixel values from the original image to the corresponding position of the new matrix.
for k in range(0,h):
	for l in range(0,w):
		img_reshape[k+center_poc,l+center_poc]=img_blurred.item(k,l)

# do the blurring process
for i in range(0,h):
	for j in range(0,w):
		sum=0
		for h1 in range(0,len(Gaussian)):
			for w1 in range(0,len(Gaussian)):
				pixel_bg=img_reshape.item(i+h1,j+w1)
				pixel_kernel=Gaussian[h1,w1]
				sum=sum+(pixel_bg*pixel_kernel) 
		img_blurred.itemset((i,j),sum)

cv2.imwrite('sydney_blurred_modified.jpg',img_blurred)

cv2.imshow('gray_image',img)
cv2.imshow('blurred_image',img_blurred)
cv2.waitKey(0)
