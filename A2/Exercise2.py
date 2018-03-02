import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.image as mping
from scipy.fftpack import dct,idct
import cv2
import math

# define the function to do dct and idct in 2D space
def dct2(image):
	return dct(dct(image.T,norm='ortho').T,norm='ortho')

def idct2(dctmatrix):
	return idct(idct(dctmatrix.T,norm='ortho').T,norm='ortho')


# read in the image and show it
img=cv2.imread('sydney.jpg',cv2.IMREAD_GRAYSCALE)
gray_img=cv2.imwrite('gray_sydney.jpg',img)


# do the dct to transfer the image from spatial space into frequency space
dct_img=dct2(img)
m,n=dct_img.shape

t_dct_img=np.zeros(shape=(m,n))

for i in range(0,m):
	for j in range(0,n):
		t_dct_img[i,j]=math.log(abs(dct_img[i,j]))


# apply the low pass filter
low_filter=dct_img
low_filter[40:,40:]=0
low_idct=idct2(low_filter)
diff_low=abs(low_idct-img)
cv2.imwrite('low_filtered_sydney.jpg',low_idct)
cv2.imwrite('diff_low.jpg',diff_low)


# apply the high pass filter
high_filter=dct_img
high_filter[0:40,0:40]=0
high_idct=idct2(high_filter)
diff_high=abs(high_idct-img)
cv2.imwrite('high_filtered_sydney.jpg',high_idct)
cv2.imwrite('diff_high.jpg',diff_high)


# show the original image
original_img=cv2.imread('gray_sydney.jpg')
original_img=cv2.cvtColor(original_img,cv2.COLOR_BGR2RGB)
plt.imshow(original_img)
plt.show()


# show the processed image by the low pass filter
low_filtered_sydney=cv2.imread('low_filtered_sydney.jpg')
low_filtered_sydney=cv2.cvtColor(low_filtered_sydney,cv2.COLOR_BGR2RGB)
plt.imshow(low_filtered_sydney)
plt.show()
# show the difference between the orignal image and processed image applied low pass filter
diff_low=cv2.imread('diff_low.jpg')
diff_low=cv2.cvtColor(diff_low,cv2.COLOR_BGR2RGB)
plt.imshow(diff_low)
plt.show()


# show the processed image by the high pass filter
high_filtered_sydney=cv2.imread('high_filtered_sydney.jpg')
high_filtered_sydney=cv2.cvtColor(high_filtered_sydney,cv2.COLOR_BGR2RGB)
plt.imshow(high_filtered_sydney)
plt.show()
# show the difference between the orignal image and processed image applied high pass filter
diff_high=cv2.imread('diff_high.jpg')
diff_high=cv2.cvtColor(diff_high,cv2.COLOR_BGR2RGB)
plt.imshow(diff_high)
plt.show()
