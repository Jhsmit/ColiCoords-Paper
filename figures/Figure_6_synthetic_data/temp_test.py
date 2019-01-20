from colicoords import load, save
import numpy as np

cells = load('temp_cells.hdf5')
names = list([c.name + '\n' for c in cells])


i = cells.name.tolist().index('Cell_13')
print(i)

with open('names.txt', 'w') as f:
    f.writelines(names)
#
# print(int_inner[0])
#
#
# check_unique(int_inner)
# check_unique(int_outer)
# check_unique(len_inner)
# check_unique(len_outer)
