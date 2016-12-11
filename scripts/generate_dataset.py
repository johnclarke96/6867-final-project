import numpy as np
from os import listdir
from os.path import isfile, join
import useful_constants as uc

size = 10
resolution = 1
lattice_side = (size/resolution)*2 + 1
input_directory = '../data/processed_pdb_files/'
output_directory = '../data/datasets/'
mutant_list = open('../data/imutant_dataset.txt', 'r')
mutant_list.readline()
mutant_list.readline()

processed_files = [f for f in listdir(input_directory) if isfile(join(input_directory, f))]

def text_to_number(text):
  split = text.split('.')
  return int(split[0])

processed_files.sort(key=text_to_number)

X1 = None
X2 = None
Y_ddg = None
Y_class = None
for filename in processed_files:
  f = open(input_directory + filename, 'r')
  f.readline()
  lattice = np.zeros((lattice_side, lattice_side, lattice_side))
  mutation = f.readline().strip()
  center_of_mass = f.readline().split()
  origin = np.array([float(center_of_mass[0]), float(center_of_mass[1]), float(center_of_mass[2])])
  for line in f:
    split_line = line.split()
    atom_code = uc.atom_codes[split_line[2]]
    coord = np.array([float(split_line[10]), float(split_line[11]), float(split_line[12])])
    new_coord = np.rint((coord - origin)).astype(int)
    lattice[new_coord] = atom_code
  if type(X1) is np.ndarray:
    X1 = np.concatenate((X1, lattice[np.newaxis, :, :, :]), axis=0)
  else:
    X1 = lattice[np.newaxis, :, :, :]
  
  old_residue_index = uc.amino_acids_index[mutation[0]]
  new_residue_index = uc.amino_acids_index[mutation[-1]]
  mutation_vector = np.zeros((1, len(uc.amino_acids_index)))
  mutation_vector[0, old_residue_index] = -1
  mutation_vector[0, new_residue_index] = 1
  if type(X2) is np.ndarray:
    X2 = np.concatenate((X2, mutation_vector), axis=0)
  else:
    X2 = mutation_vector
  
  data_line = mutant_list.readline().split()
  ddg = float(data_line[5])
  ddg_vector = np.array([ddg])
  if type(Y_ddg) is np.ndarray:
    Y_ddg = np.concatenate((Y_ddg, ddg_vector[np.newaxis, :]), axis=0)
  else:
    Y_ddg = ddg_vector[np.newaxis, :]
  
  ddg_class = 1 if ddg > 0 else 0
  ddg_class_vector = np.array([ddg_class])
  if type(Y_class) is np.ndarray:
    Y_class = np.concatenate((Y_class, ddg_class_vector[np.newaxis, :]), axis=0)
  else:
    Y_class = ddg_class_vector[np.newaxis, :]

print(X1.shape)
print(X2.shape)
print(Y_ddg.shape)
print(Y_class.shape)

np.save(output_directory + 'X_lattices', X1)
np.save(output_directory + 'X_mutations', X2)
np.save(output_directory + 'Y_ddg', Y_ddg)
np.save(output_directory + 'Y_class', Y_class)


