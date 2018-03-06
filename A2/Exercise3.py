import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import cv2


img=cv2.imread('sydney.jpg',cv2.IMREAD_GRAYSCALE)
img_blurred=img.copy()
h,w=img.shape[:2]


# set the Gaussian filter
Gaussian=1/256.*np.matrix('1;4;6;4;1')*np.matrix('1 4 6 4 1')


# pad the image and the kernel. 
# Their dimensions should be the same and at least (M(img)+m(kernel)-1)*(N(img)+n(kernel)-1)
kernel_pad_m=h-1
kernel_pad_n=w-1
img_pad_width=len(Gaussian)-1

kernel_pad=np.pad(Gaussian,((0,kernel_pad_m),(0,kernel_pad_n)),'constant')
img_pad=np.pad(img,((0,img_pad_width),(0,img_pad_width)),'constant')
# do fourier transformation 
img_frq=fftpack.fft2(img_pad)
kernel_frq=fftpack.fft2(kernel_pad)

# compute the production of kernel_frq and img_frq
pro_frq= np.multiply(img_frq,kernel_frq)

# inverse the fourier transformation
img_processed=abs(fftpack.ifft2(pro_frq))

#show processed image
cv2.imwrite('sydney_fft.jpg',img_processed)
img_fft=cv2.imread('sydney_fft.jpg')
img_fft1=cv2.cvtColor(img_fft,cv2.COLOR_BGR2RGB)
plt.imshow(img_fft1)
plt.show()

###################################################################################################
###################################################################################################

# traditional convolution
img_blurred=img.copy()

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

cv2.imwrite('sydney_blurred.jpg',img_blurred)
img_trad=cv2.imread('sydney_blurred.jpg')
img_trad1=cv2.cvtColor(img_trad,cv2.COLOR_BGR2RGB)
plt.imshow(img_trad1)
plt.show()

#####################################################################################################
#####################################################################################################

# show the difference between fft and traditional approach doing convolution
img_trad=img_blurred
img_fft=img_processed
img_trad=np.pad(img_trad,((0,img_pad_width),(0,img_pad_width)),'constant')
fft_trad_diff=img_trad-img_fft
cv2.imwrite('diff_fft_trad.jpg',fft_trad_diff)
fft_trad_diff=cv2.imread('diff_fft_trad.jpg')
fft_trad_diff=cv2.cvtColor(fft_trad_diff,cv2.COLOR_BGR2RGB)
plt.imshow(fft_trad_diff)
plt.show()

#####################################################################################################
#####################################################################################################
plt.figure(figsize=(10,4))
plt.subplot(1,3,1)
plt.imshow(img_fft1)
plt.title('FFT')
plt.subplot(1,3,2)
plt.imshow(img_trad1)
plt.title('Summation')
plt.subplot(1,3,3)
plt.imshow(fft_trad_diff)
plt.title('Difference')
plt.show()
