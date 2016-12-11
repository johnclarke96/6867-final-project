# Generates a list of all PDB file codes

from sets import Set

f = open("../data/imutant_dataset.txt", 'r')
fw = open("../data/pdb_list.txt", 'w')

pdb_codes = Set()
f.readline()
f.readline()
for line in f:
  pdb_code = line.split()[0]
  pdb_codes.add(pdb_code)

for code in pdb_codes:
  fw.write(code + "\n")
