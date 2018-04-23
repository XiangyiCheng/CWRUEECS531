import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def define_3Dpoints():
	colors=np.random.rand(27)
	fig=plt.figure()
	ax=fig.add_subplot(111,projection='3d')
	x=[-0.5, 0, 0.5]
	y=[-0.5, 0, 0.5]
	z=[-0.5, 0, 0.5]
	X, Y, Z = np.meshgrid(x, y, z)

	ax.scatter(X,Y,Z,marker='.',c=colors)

	ax.set_xlabel('X label')
	ax.set_ylabel('y label')
	ax.set_zlabel('z label')
	ax.set_xlim([-3,4])
	ax.set_ylim([-3,3])
	ax.set_zlim([-2,3])

	return ax


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
	extrinsic_matrix=np.vstack([rotation_matrix,cam_pos])

	return extrinsic_matrix

	
def intrinsic_matrix(camera):
	_,_,_,_,focal_length,film_height,film_width,width,height=camera_specification()
	cx= 0.5* (width +1)
	cy= 0.5* (height+1)

	fx= focal_length* width /film_width
	fy= focal_length* height/film_height

	intrinsic_matrix= [[fx,0,0],[0,fy,0],[cx,cy,1]]

	return intrinsic_matrix

def camera_matrix(extrinsic_matrix,intrinsic_matrix):
	camera_matrix= np.dot(extrinsic_matrix, intrinsic_matrix)
	
	return camera_matrix

camera=1
extrinsic_matrix=extrinsic_matrix(camera)
intrinsic_matrix=intrinsic_matrix(camera)
camera_matrix=camera_matrix(extrinsic_matrix,intrinsic_matrix)


