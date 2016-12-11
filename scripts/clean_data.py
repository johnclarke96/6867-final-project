f = open("../data/imutant_dataset_dirty.txt", 'r')
fw = open("../data/imutant_dataset.txt", 'w')
counter = 1
for line in f:
  fw.write(line.strip() + " ")
  if counter == 8:
    counter = 1
    fw.write("\n")
  else:
    counter += 1
