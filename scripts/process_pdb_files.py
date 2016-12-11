# processes PDB file to give input grid
import numpy as np

atomic_weights = {
  'C': 12,
  'O': 16,
  'H': 1,
  'N': 14,
  'S': 32,
}

amino_acids = {
  'A': 'ala',
  'R': 'arg',
  'N': 'asn',
  'D': 'asp',
  'B': 'asx',
  'C': 'cys',
  'E': 'glu',
  'Q': 'gln',
  'Z': 'glx',
  'G': 'gly',
  'H': 'his',
  'I': 'ile',
  'L': 'leu',
  'K': 'lys',
  'M': 'met',
  'F': 'phe',
  'P': 'pro',
  'S': 'ser',
  'T': 'thr',
  'W': 'trp',
  'Y': 'tyr',
  'V': 'val',
}

dataset = open("../data/imutant_dataset.txt", 'r')
distance_threshold = 10

def process_all_mutations(dataset, distance_threshold):
  dataset.readline()
  dataset.readline()
  counter = 1
  for line in dataset:
    split_line = line.split()
    pdb_code = split_line[0].lower()
    pdb_file = open("../data/pdb_files/" + pdb_code + ".cif", 'r')
    processed_pdb = open("../data/processed_pdb_files/" + str(counter) + ".txt", 'w')
    mutation = split_line[1]
    process_mutation(pdb_file, processed_pdb, mutation, pdb_code, distance_threshold, counter)
    pdb_file.close()
    processed_pdb.close()
    counter += 1


def process_mutation(pdb_file, processed_pdb, mutation, pdb_code, distance_threshold, counter):
  processed_pdb.write(pdb_code + '\n')
  old_residue = amino_acids[mutation[0]].upper()
  mutation_residue = int(mutation[1:-1])
  processed_pdb.write(mutation + "\n")
  if pdb_code == '1otr' or pdb_code == '1rn1':
    chain = 'B'
  elif pdb_code == '1tup':
    chain = 'C'
  else:
    chain = 'A'
  
  seek = True
  total_mass = 0
  weighted_sum = np.array([0, 0, 0], dtype='float64')
  for line in pdb_file:
    split_line = line.split()
    if split_line and split_line[0] == 'ATOM' and split_line[6] == chain and int(split_line[21]) == mutation_residue:
      seek = False
      if split_line[5] != old_residue:
        print("error!- wrong residue in %d" % counter)
        return
      coord = np.array([float(split_line[10]), float(split_line[11]), float(split_line[12])])
      mass = atomic_weights[split_line[2]]
      total_mass += mass
      weighted_sum += mass * coord
    else:
      if not seek:
        break

  if seek:
    print("error!- residue not found in %d" % counter)
    return

  center_mass = weighted_sum / total_mass
  processed_pdb.write(str(center_mass[0]) + " " + str(center_mass[1]) + " " + str(center_mass[2]) + '\n')

  pdb_file.seek(0)
  seek = True
  for line in pdb_file:
    split_line = line.split()
    if split_line and split_line[0] == 'ATOM' and split_line[6] == chain:
      seek = False
      coord = np.array([float(split_line[10]), float(split_line[11]), float(split_line[12])])
      distance = np.linalg.norm(coord - center_mass)
      if distance <= distance_threshold:
        processed_pdb.write(line)
    else:
      if not seek:
        break

if __name__ == "__main__":
  process_all_mutations(dataset, distance_threshold)
