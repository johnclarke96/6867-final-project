f = open('../data/imutant_dataset.txt', 'r')

prev_entries = []
f.readline()
f.readline()
for line in f:
  split_line = line.split()
  if not prev_entries:
    prev_entries.append(split_line)
    continue
  else:
    if split_line[0] == prev_entries[0][0] and split_line[1] == prev_entries[0][1]:
      prev_entries.append(split_line)
      continue
    else:
#to be continued (average identical mutations)

