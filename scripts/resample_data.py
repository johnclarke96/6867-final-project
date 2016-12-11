import numpy as np
import random

directory = '../data/datasets/'
X_lattices = np.load(directory + 'X_lattices.npy')
X_mutations = np.load(directory + 'X_mutations.npy')
Y_class = np.load(directory + 'Y_class.npy')
Y_ddg = np.load(directory + 'Y_ddg.npy')

increase_indices = np.where(Y_class == 1)
increase_indices_list = increase_indices[0].tolist()
increases = X_lattices[increase_indices_list, :, :, :]
resampled_X_lattices = np.concatenate([X_lattices, increases], axis=0)
increases = X_mutations[increase_indices_list, :]
resampled_X_mutations = np.concatenate([X_mutations, increases], axis=0)
increases = Y_class[increase_indices_list, :]
resampled_Y_class = np.concatenate([Y_class, increases], axis=0)
increases = Y_ddg[increase_indices_list, :]
resampled_Y_ddg = np.concatenate([Y_ddg, increases], axis=0)

np.save(directory + 'resampled_X_lattices', resampled_X_lattices)
np.save(directory + 'resampled_X_mutations', resampled_X_mutations)
np.save(directory + 'resampled_Y_class', resampled_Y_class)
np.save(directory + 'resampled_Y_ddg', resampled_Y_ddg)
