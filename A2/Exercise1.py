import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

ranm=np.random.rand(16,16)
dct=fftpack.dct(fftpack.dct(ranm.T, norm='ortho').T,norm='ortho')
plt.matshow(dct)

plt.show()

