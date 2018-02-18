import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import spline
import cv2

roc=cv2.imread('ROC.jpg')

x=(0,0,0,0,0,0,0.04,0.5,0.9,1)
y=(0.875,0.875,0.875,0.875,1,1,1,1,1,1)

x_n1=(0,0,0,0,0,0,0.039,0.5,0.9,1)
y_n1=(0.25,0.75,0.75,0.75,0.875,1,1,1,1,1)

x_n2=(0,0,0,0,0,0,0,0,0.5,1)
y_n2=(0,0,0,0,0,0,0.125,0.75,0.875,1)

plt.plot(x,y,'r',x_n1,y_n1,'b',x_n2,y_n2,'g')
f=plt.gca()
plt.xlim(-0.99,1.1)
plt.ylim(0,1.1)
plt.title('ROC curve')

plt.show()
cv2.imshow('roc',roc)
cv2.waitKey()
#auc=np.trapz(y,x)

