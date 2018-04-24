import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def define_3Dpoints():
	
	x=[-0.5, 0, 0.5]
	y=[-0.5, 0, 0.5]
	z=[-0.5, 0, 0.5]
	X, Z, Y = np.meshgrid(x, y, z)

	dimension=X.shape[0]*Y.shape[0]*Z.shape[0]
	
	colors=np.zeros((dimension,3))
	for i in range (0, dimension):
		colors[i,:]=np.random.rand(3)

	return X,Y,Z,colors


def camera_specification():
	# position parameters
	r= 5
	alpha1= np.pi/6
	beta= np.pi/6

	# camera 1 parameters
	cam1_pos= [r*np.cos(beta)*np.cos(alpha1), r*np.cos(beta)*np.sin(alpha1), r*np.sin(beta)]
	target= np.array([0,0,0])
	up= np.array([0,0,1])
	focal_length= 0.06
	film_width= 0.035
	film_height= 0.035
	width= 256
	height= 256	

	# camera 2 parameters, others are the same as the camera 1 
	alpha2= np.pi/3
	cam2_pos= [r*np.cos(beta)*np.cos(alpha2), r*np.cos(beta)*np.sin(alpha2), r*np.sin(beta)]
	return cam1_pos,cam2_pos,target,up,focal_length,film_height,film_width,width,height


def camera_view_unit(target,cam_pos,up):
	zcam= target-cam_pos
	xcam= np.cross(zcam,up)
	ycam= np.cross(zcam,xcam)

	# normalization
	if np.linalg.norm(xcam)!=0:
		xcam= xcam/np.linalg.norm(xcam)
	
	if np.linalg.norm(ycam)!=0:
		ycam= ycam/np.linalg.norm(ycam)
	
	if np.linalg.norm(zcam)!=0:
		zcam= zcam/np.linalg.norm(zcam)
	
	return xcam,ycam,zcam


def extrinsic_matrix(camera):
	cam1_pos,cam2_pos,target,up,_,_,_,_,_=camera_specification()

	if camera==1:
		cam_pos=cam1_pos
	elif camera==2:
		cam_pos=cam2_pos
	else:
		print 'camera is not defined.'

	xcam,ycam,zcam=camera_view_unit(target,cam_pos,up)

	rotation_matrix=np.column_stack((xcam,ycam,zcam))
	add=np.dot(np.dot(-1,cam_pos),rotation_matrix)

	extrinsic_matrix=np.vstack([rotation_matrix,add])

	return extrinsic_matrix

	
def intrinsic_matrix(camera):
	_,_,_,_,focal_length,film_height,film_width,width,height=camera_specification()
	cx= 0.5* (width +1)
	cy= 0.5* (height+1)

	fx= focal_length* width /film_width
	fy= focal_length* height/film_height

	intrinsic_matrix= [[fx,0,0],[0,fy,0],[cx,cy,1]]

	return intrinsic_matrix

def camera_matrix(camera,extrinsic_matrix,intrinsic_matrix):
	extrinsic_matrix=extrinsic_matrix(camera)
	intrinsic_matrix=intrinsic_matrix(camera)
	camera_matrix= np.dot(extrinsic_matrix, intrinsic_matrix)

	return camera_matrix


def conv_2Dimage(camera,camera_matrix):
	X,Y,Z,colors=define_3Dpoints()
	X_reshape=X.reshape(1,-1)
	Y_reshape=Y.reshape(1,-1)
	Z_reshape=Z.reshape(1,-1)
	_,dimension=X_reshape.shape

	point=np.column_stack((X_reshape,Y_reshape,Z_reshape,np.ones((1,dimension))))
	point_reshape=np.transpose(point.reshape(4,-1))
	pt=np.dot(point_reshape,camera_matrix)
	object_x=np.transpose(pt[:,0]/pt[:,2])
	object_y=np.transpose(pt[:,1]/pt[:,2])

	#object_2D=np.column_stack((object_x,object_y))

	return object_x,object_y,colors


def plot_2Dimage(object_x,object_y,colors):
	_,_,_,_,_,_,_,width,height=camera_specification()
	color_matrix= np.zeros((height,width,3))
	object_number=object_x.shape[0]
	show_region=5

	for i in range (0,object_number):
		r1= int(max(1,np.floor(object_y[i]-show_region))) 
		r2= int(min(height,np.ceil(object_y[i]+show_region)))
		c1= int(max(1,np.floor(object_x[i]-show_region)))
		c2= int(min(width,np.ceil(object_x[i]+show_region)))

		for r in range (r1,r2+1):
			for c in range (c1,c2+1):
				if (r-object_y[i])**2+(c-object_x[i])**2< show_region**2:
					color_matrix[r,c,:]=colors[i,:]
	
					
	plt.imshow(color_matrix)
	plt.show()



camera=1
camera_matrix=camera_matrix(camera,extrinsic_matrix,intrinsic_matrix)
object_x,object_y,colors=conv_2Dimage(camera,camera_matrix)
plot_2Dimage(object_x,object_y,colors)

