from scipy.signal import correlate2d
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.fftpack import dct,idct

seq1 = {'I1': io.imread('data/image/seq1/frame1.png', as_grey=True), 
        'I2': io.imread('data/image/seq1/frame3.png', as_grey=True),
        'U' : np.loadtxt('data/flow/seq1/flow3.u', dtype='double', delimiter=','),
        'V' : np.loadtxt('data/flow/seq1/flow3.v', dtype='double', delimiter=',')}

rubic = {'I1':io.imread('data/rubic/rubic.0.png', as_grey=True), 
         'I2':io.imread('data/rubic/rubic.5.png', as_grey=True)}

sphere= {'I1': io.imread('data/sphere/sphere.1.png', as_grey=True), 
         'I2': io.imread('data/sphere/sphere.3.png', as_grey=True)}

def dct2(image):
    return dct(dct(image.T,norm='ortho').T,norm='ortho')

def idct2(dctmatrix):
    return idct(idct(dctmatrix.T,norm='ortho').T,norm='ortho')

def correlation_each_grid(I1, I2, x, y, width, region_expand):
    h= I1.shape[0]
    w= I1.shape[1]

    x = int(x)
    y = int(y)

    half_width = int(np.floor(width/2))

    # patch construction
    if (y-half_width)> 0:
        p_x_up=y-half_width
    else:
        p_x_up=0

    if (y+half_width+1) < h:
        p_x_down=y+half_width+1
    else:
        p_x_down=h

    if (x-half_width)> 0:
        p_y_left=x-half_width
    else:
        p_y_left=0

    if (x+half_width+1) < w:
        p_y_right=x+half_width+1
    else:
        p_y_right=w
    

    dct_img=dct2(I1)
    patch = dct_img[p_x_up:p_x_down, p_y_left:p_y_right]
    
    # correlation region
    if y-half_width-region_expand> 0:
        r_x_up=y-half_width+region_expand
    else:
        r_x_up=0

    if y+half_width+1+region_expand < h:
        r_x_down=y+half_width+1+region_expand
    else:
        r_x_down=h

    if x-half_width-region_expand > 0:
        r_y_left = x-half_width+region_expand
    else:
        r_y_left=0

    if x+half_width+1+region_expand < w:
        r_y_right=x+half_width+1+region_expand
    else:
        r_y_right=w

    dct_region=dct2(I2)
    region = dct_region[r_x_up:r_x_down, r_y_left:r_y_right]
    
    # correlation
    #correlation = correlate2d(region, patch, 'valid')
    correlation=idct(patch*np.conjugate(region))

  optical_flow_estimation(sphere)
    i, j = np.unravel_index(correlation.argmax(), correlation.shape)
    u = j+r_y_left-p_y_left
    v = i+r_x_up-p_x_up

    return (u, v)



def quiver_drawing(I, x_grid, y_grid, U, V, scale):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=80)
    ax.imshow(I, cmap='gray')
    ax.quiver(x_grid, y_grid, U*scale, V*scale, color='red', angles='xy', scale_units='xy', scale=1)
    ax.set_aspect('equal')
    plt.show()



def optical_flow_estimation(img_series):
    h_img= img_series['I1'].shape[0]
    w_img= img_series['I1'].shape[1]

    grid_size = 10
    width  = 5

    x = np.arange(0, w_img-grid_size, grid_size) + np.floor(grid_size/2);
    y = np.arange(0, h_img-grid_size, grid_size) + np.floor(grid_size/2);
    x_grid, y_grid = np.meshgrid(x,y);

    region_expand = 0

    h_grid = x_grid.shape[0]
    w_grid = x_grid.shape[1]

    U = np.zeros((h_grid, w_grid))
    V = np.zeros((h_grid, w_grid))
    
    for i in range(0, h_grid):
        for j in range(0, w_grid):
            u,v =  correlation_each_grid(img_series['I1'], img_series['I2'],x_grid[i, j], y_grid[i, j], width, region_expand)
            U[i, j] = u
            V[i, j] = v


    quiver_drawing(img_series['I1'],x_grid, y_grid, U, V, 1)
    
optical_flow_estimation(sphere)
