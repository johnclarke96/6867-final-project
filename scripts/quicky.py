import numpy as np

Y_class = np.load('../data/datasets/Y_class.npy')

increases = np.count_nonzero(Y_class)
total = Y_class.shape[0]
print float(increases) / total
