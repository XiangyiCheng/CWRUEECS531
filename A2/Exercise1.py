import numpy as np
import matplotlib.pyplot as plt
import math

def dctbasis(u,v,m,n):
	# set the constant value of cu and cv
	if u==0:
		cu=1.0/math.sqrt(m)
	else:
		cu=math.sqrt(2.0/m)

	if v==0:
		cv=1.0/math.sqrt(n)
	else:
		cv=math.sqrt(2.0/n)
	

	# initialize the matrix of each basis
	dctb=np.zeros(shape=(m,n))

	# fill values in each basis 
	for i in range(0,m): # m times iteration
		for j in range(0,n): # n times iteration
			dctb[i,j]=cu*cv*math.cos(math.pi*(2*i+1)*u/(2*m))*math.cos(math.pi*(2*j+1)*v/(2*n))

	return dctb



# main function
m=input("what is the value of dimension m?")
n=input("what is the value of dimension n?")
u=m
v=n

basis=np.zeros(shape=(m*n,m*n))

for u in range(0,m):
	for v in range(0,n):
		onebasis=dctbasis(u,v,m,n)
		basis[u*m:(u+1)*m,v*n:(v+1)*n]=onebasis[:,:]


#basis=ListedColormap(colormap.colors[::-1])
plt.imshow(basis,cmap='gist_gray')
plt.show()
plt.imshow(basis)
plt.show()


