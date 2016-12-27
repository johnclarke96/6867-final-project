import numpy as np
import useful_constants as uc

X_lattices = np.load('../data/datasets/resampled_X_lattices.npy')
number_atoms = len(uc.atom_codes)
channels = []

for _, number in uc.atom_codes.items():
  channel = np.where(X_lattices == number, 1, 0)[:, :, :, :, np.newaxis]
  channels.append(channel)

multichannel_X_lattices = np.concatenate(channels, axis=4)
np.save('../data/datasets/resampled_multichannel_X_lattices', multichannel_X_lattices)
