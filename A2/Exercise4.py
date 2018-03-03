from mnist import MNIST
import random
import matplotlib.pyplot as plt
import numpy as np

mndata = MNIST('/home/liutao/python-mnist/data')
images, labels = mndata.load_training()
#index = random.randrange(0,len(images))
#print mndata.display(images[index])
first_image=images[0]
first_image=np.array(first_image,dtype='float')
pixels=first_image.reshape((28,28))
plt.imshow(pixels,cmap='gray')
plt.show()

