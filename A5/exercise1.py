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


def plot_camera(ax,camera):
	cam1_pos,cam2_pos,target,up,focal_length,film_height,film_width,width,height= camera_specification()
	if camera==1:
		cam_pos=cam1_pos
		color='r'
	elif camera==2:
		cam_pos=cam2_pos
		color='b'
	else:
		print 'the camera is not defined.'
	
	ax.scatter(cam_pos[0],cam_pos[1],cam_pos[2],marker='.',c=color)
	
	xcam,ycam,zcam= camera_view_unit(target,cam_pos,up)
	# four corners on the plane through focal points
	d= np.linalg.norm(target-cam_pos)

	x= 0.5* film_width*d/focal_length
	y= 0.5* film_height*d/focal_length

	p1 = cam_pos + x* xcam + y* ycam + d* zcam;
	p2 = cam_pos + x* xcam - y* ycam + d* zcam;
	p3 = cam_pos - x* xcam - y* ycam + d* zcam;
	p4 = cam_pos - x* xcam + y* ycam + d* zcam;

	# connect these 4 points
	plotx=[p1[0],p2[0],p3[0],p4[0],p1[0]]
	ploty=[p1[1],p2[1],p3[1],p4[1],p1[1]]
	plotz=[p1[2],p2[2],p3[2],p4[2],p1[2]]
	#ax.plot(plotx,ploty,plotz,c=color,linewidth=0.5)
	verts=[zip(plotx,ploty,plotz)]
	face=Poly3DCollection(verts,alpha=0.2,facecolors=color,linewidth=0.5)
	#ax.add_collection3d(Poly3DCollection(verts,alpha=0.5,facecolors=color))
	ax.add_collection3d(face)
	alpha=0.2
	if camera==1:
		face.set_facecolor((1,0,0,alpha))
	else:
		face.set_facecolor((0,0,1,alpha))

	ax.plot([cam_pos[0],target[0]],[cam_pos[1],target[1]],[cam_pos[2],target[2]],c=color,linewidth=0.3)

	ax.plot([cam_pos[0],p1[0]],[cam_pos[1],p1[1]],[cam_pos[2],p1[2]],c=color,linewidth=0.3)
	ax.plot([cam_pos[0],p2[0]],[cam_pos[1],p2[1]],[cam_pos[2],p2[2]],c=color,linewidth=0.3)
	ax.plot([cam_pos[0],p3[0]],[cam_pos[1],p3[1]],[cam_pos[2],p3[2]],c=color,linewidth=0.3)
	ax.plot([cam_pos[0],p4[0]],[cam_pos[1],p4[1]],[cam_pos[2],p4[2]],c=color,linewidth=0.3)


ax=define_3Dpoints()
plot_camera(ax,1)
plot_camera(ax,2)
#plt.show()
#ax.view_init(elev=35,azim=65)
#ax.view_init(elev=30,azim=30)

plt.show()
#camera_specification()
