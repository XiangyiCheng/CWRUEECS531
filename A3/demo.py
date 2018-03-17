import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from keras.datasets import mnist

# download data from internet (only the first time) and split it into training and testing set
(imgTrain, labelTrain), (imgTest, labelTest) = mnist.load_data()

from keras import backend as K

imgRows, imgCols = 28, 28 # size of source image

# reshape images in training and testing set into fake 3D
if K.image_data_format() == 'channels_first':
    imgTrain = imgTrain.reshape(imgTrain.shape[0], 1, imgRows, imgCols)
    imgTest  = imgTest.reshape(imgTest.shape[0], 1, imgRows, imgCols)
    smpSize  = (1, imgRows, imgCols)
else:
    imgTrain = imgTrain.reshape(imgTrain.shape[0], imgRows, imgCols, 1)
    imgTest  = imgTest.reshape(imgTest.shape[0], imgRows, imgCols, 1)
    smpSize  = (imgRows, imgCols, 1)

# convert pixels to floats and map them into range of [0,1]
imgTrain = imgTrain.astype('float') / 255
imgTest  = imgTest.astype('float') / 255

# show shape and type of our datasets
print('Training set in shape of ', imgTrain.shape, ' with element type ', type(imgTrain.item(0)))
print('Testing set in shape of  ', imgTest.shape, ' with element type ', type(imgTrain.item(0)))

ncat = 10 # number of categories in our problem

# convert labels to one-hot vectors
onehotTrain = keras.utils.to_categorical(labelTrain, ncat)
onehotTest  = keras.utils.to_categorical(labelTest, ncat)

# claim a sequential model
model = Sequential()

# add first layer as convolution transformation with 32 filters of size [3, 3]
# - 'activation' is the non-linear tranform used in neural network, a common one is 'sigmoid'
# - for the first layer, you also need to claim size of input samples
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=smpSize))
# add max-pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
# add flatten layer to reshape data to fit for linear transform
model.add(Flatten())
# linear transofrm with activation of softmax
model.add(Dense(ncat, activation='softmax'))


# compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(imgTrain, onehotTrain, validation_data=(imgTest, onehotTest), batch_size=128, epochs=3, verbose=1)

score = model.evaluate(imgTest, onehotTest, verbose=0)
print('Test loss     :', score[0])
print('Test accuracy :', score[1])
