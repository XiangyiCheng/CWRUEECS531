from scipy.signal import correlate2d
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

seq1 = {'I1': io.imread('data/image/seq1/frame1.png', as_grey=True), 
        'I2': io.imread('data/image/seq1/frame3.png', as_grey=True),
        'U' : np.loadtxt('data/flow/seq1/flow3.u', dtype='double', delimiter=','),
        'V' : np.loadtxt('data/flow/seq1/flow3.v', dtype='double', delimiter=',')}

rubic = {'I1':io.imread('data/rubic/rubic.0.png', as_grey=True), 
         'I2':io.imread('data/rubic/rubic.5.png', as_grey=True)}

sphere= {'I1': io.imread('data/sphere/sphere.1.png', as_grey=True), 
         'I2': io.imread('data/sphere/sphere.3.png', as_grey=True)}

def quiver_drawing(I, X, Y, U, V, scale):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=80)
    ax.imshow(I, cmap='gray')
    ax.quiver(X, Y, U*scale, V*scale, color='red', angles='xy', scale_units='xy', scale=1)
    ax.set_aspect('equal')
    plt.show()

def correlation_each_grid(I1, I2, x, y, n, u0, v0):
    h, w = I1.shape
    x = int(x)
    y = int(y)
    n2 = int(np.floor(n/2))
    # extract patch in I1 at x, y
    if (y-n2)> 0:
        i11=y-n2
    else:
        i11=0

    if (y+n2+1) < h:
        i12=y+n2+1
    else:
        i12=h

    if (x-n2)> 0:
        j11=x-n2
    else:
        j11=0

    if (x+n2+1) < w:
        j12=x+n2+1
    else:
        j12=w

    patch = I1[i11:i12, j11:j12]
    
    # extract the search region in I2
    if y-n2-v0> 0:
        i21=y-n2-v0
    else:
        i21=0

    if y+n2+1+v0 < h:
        i22=y+n2+1+v0
    else:
        i22=h

    if x-n2-u0 > 0:
        j21 = x-n2-u0
    else:
        j21=0

    if x+n2+1+u0 < w:
        j22=x+n2+1+u0
    else:
        j22=w

    region = I2[i21:i22, j21:j22]
    
    # correlation
    response = correlate2d(region, patch, 'valid')
    i, j = np.unravel_index(response.argmax(), response.shape)
    
    # caculate mv
    u = j21-j11+j
    v = i21-i11+i
    return (u, v)


def optical_flow_estimation(seq):
    h_img= seq['I1'].shape[0]
    w_img= seq['I1'].shape[1]

    grid_size = 5
    width  = 5

    x = np.arange(0, w_img-grid_size, grid_size) + np.floor(grid_size/2);
    y = np.arange(0, h_img-grid_size, grid_size) + np.floor(grid_size/2);
    x_grid, y_grid = np.meshgrid(x,y);

    x_expand = 5
    y_expand = 5

    h_grid = x_grid.shape[0]
    w_grid = x_grid.shape[1]

    U = np.zeros((h_grid, w_grid))
    V = np.zeros((h_grid, w_grid))
    
    for i in range(0, h_grid):
        for j in range(0, w_grid):
            u,v =  correlation_each_grid(seq['I1'], seq['I2'],x_grid[i, j], y_grid[i, j], width, x_expand, y_expand)
            U[i, j] = u
            V[i, j] = v


    quiver_drawing(seq['I1'],x_grid, y_grid, U, V, 1)
    
optical_flow_estimation(sphere)

