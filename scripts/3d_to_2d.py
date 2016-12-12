import numpy as np

data = np.load('../data/datasets/resampled_X_lattices.npy')

def transform_2d(X_3d_lattices = data):

	X_3d_shape = X_3d_lattices.shape
	n,X,Y,Z = X_3d_shape
	assert X == Y == Z

	X_2d_xy_transform = np.zeros((n,X,Y,1))
	X_2d_yz_transform = np.zeros((n,Y,Z,1))
	X_2d_xz_transform = np.zeros((n,X,Z,1))

	# perform xy transformation
	for num in range(n):
		for i in range(X):
			for j in range(Y):
				val = 0
				for count in range(1,Z): 
					val = X_3d_lattices[num][i][j][Z-count]
					if val != 0:
						break
				X_2d_xy_transform[num][i][j][0] = val

	# perform yz transformation
	for num in range(n):
		for i in range(Y):
			for j in range(Z):
				val = 0
				for count in range(1,X): 
					val = X_3d_lattices[num][i][j][X-count]
					if val != 0:
						break
				X_2d_yz_transform[num][i][j][0] = val

	# perform xz transformation
	for num in range(n):
		for i in range(X):
			for j in range(Z):
				val = 0
				for count in range(1,Y):
					val = X_3d_lattices[num][i][j][Y-count]
					if val != 0:
						break
				X_2d_xz_transform[num][i][j][0] = val

	X_2d_lattices = np.concatenate((X_2d_xy_transform, X_2d_yz_transform, X_2d_xz_transform), axis=3)
	return X_2d_lattices


