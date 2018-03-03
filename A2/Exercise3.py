import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mping
from scipy import fftpack
import cv2
import math

def zero_pad_center(vector,pad_width):
	vector=np.pad(vector,((pad_width,pad_width),(pad_width,pad_width)),'constant')
	return vector 


img=cv2.imread('sydney.jpg',cv2.IMREAD_GRAYSCALE)
img_blurred=img.copy()
h,w=img.shape[:2]

# resize the image to a square image.
if h<w:
	img_size=h
	img_resize=cv2.resize(img,(h,h))
else:
	img_size=w
	img_resize=cv2.resize(img,(w,w))

# set the Gaussian filter
Gaussian=1/256.*np.matrix('1;4;6;4;1')*np.matrix('1 4 6 4 1')


# pad the image and the kernel. 
# Their dimensions should be the same and at least (M(img)+m(kernel)-1)*(N(img)+n(kernel)-1)
kernel_pad_width=img_size-1
img_pad_width=len(Gaussian)-1

kernel_pad=np.pad(Gaussian,((0,kernel_pad_width),(0,kernel_pad_width)),'constant')
img_pad=np.pad(img_resize,((0,img_pad_width),(0,img_pad_width)),'constant')
# do fourier transformation 
img_frq=fftpack.fft2(img_pad)
kernel_frq=fftpack.fft2(kernel_pad)

# compute the production of kernel_frq and img_frq
pro_frq= np.multiply(img_frq,kernel_frq)

# inverse the fourier transformation
img_processed=fftpack.ifft2(pro_frq)

#show processed image
cv2.imwrite('sydney_fft.jpg',abs(img_processed))
img_fft=cv2.imread('sydney_fft.jpg')
img_fft=cv2.cvtColor(img_fft,cv2.COLOR_BGR2RGB)
plt.imshow(img_fft)
plt.show()


