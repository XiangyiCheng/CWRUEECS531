from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

# load MNIST data
mndata = MNIST('/home/liutao/python-mnist/data')
images, labels = mndata.load_training()

##########################################################################################
##########################################################################################

# generate a matrix and put each image information in each row
# image size is 28*28
datam=np.zeros((len(images),28*28))
for index in range(0,len(images)):
	img=images[index]
	img=np.array(img,dtype='float')
	datam[index,]=img

# get the mean value of each column (compare images)
mean=np.mean(datam,axis=0)
# subtract mean value from the matrix and transpose it to fit in linalg.eig()
# M is based on processed image information in each column
M=(datam-mean).T

# obtain the covariance matrix of M
# latent is the eigenvalues. coeff is the normalized eigenvectors.
# do PCA in each column
[latent,coeff]=np.linalg.eig(np.cov(M))
score=np.dot(coeff.T,M)

###########################################################################################
###########################################################################################

plt.subplot(2,4,1)
plt.imshow(mean.reshape((28,28)),cmap='gray')
plt.title('mean digit')

components=coeff[:,0:7].reshape(28,28,7)
for k in range(0,7):
	eigenface=abs(components[:,:,k])
	plt.subplot(2,4,k+2)
	plt.imshow(eigenface,cmap='gray')
	plt.title('%ith eigenvector'%(k+1))
plt.show()

###########################################################################################
###########################################################################################

img_id=0
img=images[img_id]
img=np.array(img,dtype='float')
pixels=img.reshape((28,28))
plt.subplot(2,5,1)
plt.imshow(pixels,cmap='gray')
plt.title('original')


for k in range(1,5):
	# score[0:k*30,img_id] is a row vector
	# score.shape=(28*28,60000)
	reconstruction=np.dot(score[0:k*30,img_id],coeff[:,0:k*30].T)
	reconstruction_tsf=abs((reconstruction+mean).reshape((28,28)))
	diff=np.square(reconstruction_tsf-pixels)
	plt.subplot(2,5,k+1)
	plt.imshow(reconstruction_tsf,cmap='gray')
	plt.title('k=%i'%(k*30))
	plt.subplot(2,5,k+6)
	plt.imshow(diff,cmap='gray')
	plt.title('errors')
plt.show()

