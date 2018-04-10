import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

rubic_start=cv2.imread('data/rubic/rubic.0.png')
rubic_end=cv2.imread('data/rubic/rubic.5.png')

# canny edge detector, set different lower and upper thresholds for the detector.
rubic_s=cv2.Canny(rubic_start,400,400)
rubic_e=cv2.Canny(rubic_end,400,400)
cv2.imwrite('rubic_s.png',rubic_s)
cv2.imwrite('rubic_e.png',rubic_e)
#cv2.waitKey()

rubic = {'I1':io.imread('rubic_s.png', as_grey=True), 
         'I2':io.imread('rubic_e.png', as_grey=True)}

def quiver_drawing(I, x_grid, y_grid, U, V, scale):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=80)
    ax.imshow(I, cmap='gray')
    ax.quiver(x_grid, y_grid, U*scale, V*scale, color='red', angles='xy', scale_units='xy', scale=1)
    ax.set_aspect('equal')
    plt.show()



def derivative_components(I1, I2, x_grid, y_grid):
    h_img= I1.shape[0]
    w_imd= I1.shape[1]
    x_grid = int(x_grid)
    y_grid = int(y_grid)

    if x_grid>0 and x_grid< (w_imd-1) and y_grid>=0 and y_grid<h_img:
        Ix=(I1[y_grid, x_grid+1] - I1[y_grid, x_grid-1])/2;
    else:
        Ix=0

    if x_grid>=0 and x_grid<w_imd and y_grid>0 and y_grid< (h_img-1):
        Iy=(I1[y_grid+1, x_grid] - I1[y_grid-1, x_grid])/2;
    else:
        Iy=0

    if x_grid>=0 and x_grid<w_imd and y_grid>=0 and y_grid<h_img:
        It=I2[y_grid,x_grid] - I1[y_grid,x_grid]
    else:
        It=0

    return (Ix, Iy, It)



def solve_motion_gradient_equation(I1,I2,x_grid,y_grid,width,disx,disy):


    A = np.zeros((width*width, 2))
    b = np.zeros(width*width)
    


    # obtain A and b from I matrix
    for i in range(0, width*width):

        x_grid_m= x_grid+disx[i]
        y_grid_m= y_grid+disy[i]

        Ix, Iy, It = derivative_components(I1, I2, x_grid_m, y_grid_m)
        A[i, 0] = Ix 
        A[i, 1] = Iy
        b[i] = -It

    motion_matrix = np.linalg.lstsq(np.matmul(A.T, A), np.matmul(A.T, b))

    motion_estimation = motion_matrix[0]
    return motion_estimation


def estimate_flow(img_series):
    # get the image size.
    h_img = img_series['I1'].shape[0]
    w_img= img_series['I1'].shape[1]

    # define the grid size
    grid_size = 9
    width  = 21

    # x, y are the locations of the grids.
    x = np.arange(0, w_img-grid_size, grid_size) + np.floor(grid_size/2);
    y = np.arange(0, h_img-grid_size, grid_size) + np.floor(grid_size/2);

 
    x_grid, y_grid = np.meshgrid(x,y);

    
    # get the height and width of the grid
    h_grid= x_grid.shape[0]
    w_grid= x_grid.shape[1]  
   
    U = np.zeros((h_grid, w_grid))
    V = np.zeros((h_grid, w_grid))

    grid_center = np.arange(0, width) - np.floor(width/2)
    disx, disy = np.meshgrid(grid_center, grid_center)
    disx = disx.reshape(width*width, 1)
    disy = disy.reshape(width*width, 1)

    
    # get the U,V matrix
    for i in range(0, h_grid):
        for j in range(0, w_grid):
            # x_grid[i,j] is the x coordinate; y_grid[i,j] is the y coordinate.
            motion_estimation =  solve_motion_gradient_equation(img_series["I1"], img_series["I2"],x_grid[i,j], y_grid[i,j], width,disx,disy)
            U[i, j] = motion_estimation[0]
            V[i, j] = motion_estimation[1]

    quiver_drawing(img_series["I1"], x, y, U, V, 5)

estimate_flow(rubic)
