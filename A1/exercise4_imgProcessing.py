from scipy import misc
import numpy as np
import cv2

# read the image as an array. 'L'(8 pixels, black and white)
img=misc.imread('shapes.jpg',mode='L')

# img.std() calculates the standard deviation of the image.
# np.random.random((m,n)) returns m*n dimensional random floats in the interval [0.0,1.0).
noisy1=img+4*img.std()*np.random.random(img.shape)
noisy2=img+6*img.max()*np.random.random(img.shape)

# save the images with noises.
img1=cv2.imwrite('shape_noise1.jpg',noisy1)
img2=cv2.imwrite('shape_noise2.jpg',noisy2)

# count how many detections could be found.
count1=0
count2=0

# shape_noise1.jpg detection
shape_noise1=cv2.imread('shape_noise1.jpg')
shape_noise1_gray=cv2.cvtColor(shape_noise1,cv2.COLOR_BGR2GRAY)

template=cv2.imread('template.jpg',0)
w,h=template.shape[::-1] # returns all elements except for the last one.

result1=cv2.matchTemplate(shape_noise1_gray,template,cv2.TM_CCOEFF_NORMED)
threshold1=0.8
location1=np.where(result1>=threshold1)
for pt1 in zip(*location1[::-1]):
	cv2.rectangle(shape_noise1,pt1,(pt1[0]+w,pt1[1]+h),(255,255,0),1)
	count1+=1


# shape_noise2.jpg detection
shape_noise2=cv2.imread('shape_noise2.jpg')
shape_noise2_gray=cv2.cvtColor(shape_noise2,cv2.COLOR_BGR2GRAY)
result2=cv2.matchTemplate(shape_noise2_gray,template,cv2.TM_CCOEFF_NORMED)
threshold2=0.8
location2=np.where(result2>=threshold2)

for pt2 in zip(*location2[::-1]):
	cv2.rectangle(shape_noise2,pt2,(pt2[0]+w,pt2[1]+h),(255,255,0),1)
	count2+=1

print "count1=",count1,"count2=",count2

# show the detected objects on the images
cv2.imshow('shape_noise1',shape_noise1)
cv2.imshow('shape_noise2',shape_noise2)
cv2.waitKey()
