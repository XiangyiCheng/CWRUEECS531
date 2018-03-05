from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

mndata = MNIST('/home/liutao/python-mnist/data')
images, labels = mndata.load_training()

##########################################################################################
##########################################################################################

datam=np.zeros((len(images),28*28))
for index in range(0,len(images)):
	img=images[index]
	img=np.array(img,dtype='float')
	datam[index,]=img

mean=np.mean(datam,axis=0)

M=(datam-mean).T
[latent,coeff]=np.linalg.eig(np.cov(M))
score=np.dot(coeff.T,M)
#print score.shape
#print coeff.shape
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
	plt.title('%ith eigenface'%(k+1))
plt.show()

###########################################################################################
###########################################################################################

img_id=0
img=images[img_id]
img=np.array(img,dtype='float')
pixels=img.reshape((28,28))
#plt.imshow(pixels,cmap='gray')
plt.subplot(1,5,1)
plt.imshow(pixels,cmap='gray')
plt.title('original')


for k in range(1,5):
	reconstruction=np.dot(score[0:k*25,img_id],coeff[:,0:k*25].T)
	reconstruction_tsf=abs((reconstruction+mean).reshape((28,28)))
	plt.subplot(1,5,k+1)
	plt.imshow(reconstruction_tsf,cmap='gray')
	plt.title('k=%i'%(k*25))
plt.show()
