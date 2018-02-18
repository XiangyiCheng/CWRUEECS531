import cv2
import numpy as np

img=cv2.imread('sydney.jpg',cv2.IMREAD_GRAYSCALE)
img_blurred=img.copy()
# get the height and weight of the image.
h,w=img.shape[:2]

# create a Gaussian matrix.
Gaussian=1/256.*np.matrix('1;4;6;4;1')*np.matrix('1 4 6 4 1')
# calculate the iteration numbers for height and weight respectively.
h_iteration=h-len(Gaussian)+1
w_iteration=w-len(Gaussian)+1

for i in range(0,h_iteration):
	for j in range(0,w_iteration):
		sum=0
		for h1 in range(0,len(Gaussian)):
			for w1 in range(0,len(Gaussian)):
				# get the value of the pixel from the image and kernel.
				pixel_bg=img.item(i+w1,j+h1)
				pixel_kernel=Gaussian[w1,h1]
				# do the convolution.
				sum=sum+(pixel_bg*pixel_kernel) 
		# give the final value to the pixel on the processed image. 
		img_blurred.itemset((i+2,j+2),sum)

# save the blurred image.
cv2.imwrite('img_blurred.jpg',img_blurred)

# show the images.
cv2.imshow('gray_image',img)
cv2.imshow('blurred_image',img_blurred)
cv2.waitKey(0)
