import cv2
import numpy as np

# read the image in color with the RGB channel.
img=cv2.imread('sydney.jpg',cv2.IMREAD_COLOR)
img_blurred=img.copy()
h,w=img.shape[:2]

Gaussian=1/256.*np.matrix('1;4;6;4;1')*np.matrix('1 4 6 4 1')
center_poc=(len(Gaussian)-1)/2

# create a 3D zero matrix with height, weight, color channel.
img_reshape=np.zeros((h+2*center_poc,w+2*center_poc,3))

# # give the original pixel values from the original image to the corresponding position of the new matrix with color channel.
for k in range(0,h):
	for l in range(0,w):
		for d in range(0,3): # 0--R, 1--G, 2--B in color channel.
			img_reshape[k+center_poc,l+center_poc,d]=img_blurred.item(k,l,d)
	
# do the blurring process.
for c in range(0,3): # 0--R, 1--G, 2--B in color channel.
	for i in range(0,h):
		for j in range(0,w):
			sum=0
			for h1 in range(0,len(Gaussian)):
				for w1 in range(0,len(Gaussian)):
					pixel_bg=img_reshape.item(i+h1,j+w1,c)
					pixel_kernel=Gaussian[h1,w1]
					sum=sum+(pixel_bg*pixel_kernel) 
			img_blurred.itemset((i,j,c),sum)

# save the blurred image.
cv2.imwrite('sydney_blurred_colored.jpg',img_blurred)

cv2.imshow('gray_image',img)
cv2.imshow('blurred_image',img_blurred)
cv2.waitKey(0)
