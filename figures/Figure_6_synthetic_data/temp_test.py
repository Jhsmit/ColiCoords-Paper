from colicoords import load, save
import numpy as np

def check_unique(arr):
    l1 = len(arr)
    l2 = len(np.unique(arr))
    print(l1==l2, l1, l2)

gt_cells = load('cells_final_selected.hdf5')

int_inner = np.zeros(len(gt_cells))

for i, cell in enumerate(gt_cells):
    int_arr = cell.data.data_dict['storm_outer']['intensity']
    encode_arr = (np.arange(len(int_arr)) + 1)**2
    int_inner[i] = np.sum(int_arr*encode_arr)


check_unique(int_inner)


# int_inner = np.array([(c.data.data_dict['storm_outer']['intensity'] * np.arange(len.sum(c.data.data_dict['storm_outer']['intensity']))+1 for c in gt_cells])
# int_outer = np.array([c.data.data_dict['storm_inner']['intensity'].sum() for c in gt_cells])

#
#
np.save('int_inner.npy', int_inner)
# np.save('int_inner.npy', int_outer)
# np.save('len_inner.npy', len_inner)
# np.save('len_outer.npy', len_outer)
#
#
# st = np.load('storm_inner.npy')
# print(st['intensity'][:10])
# print(st)
#
# print('hallo')
#
# int_inner = np.load('int_inner.npy')
# int_outer = np.load('int_inner.npy')
# len_inner = np.load('len_inner.npy')
# len_outer = np.load('len_outer.npy')
#
# def check_unique(arr):
#     l1 = len(arr)
#     l2 = len(np.unique(arr))
#     print(l1==l2, l1, l2)
#
# print(int_inner[0])
#
#
# check_unique(int_inner)
# check_unique(int_outer)
# check_unique(len_inner)
# check_unique(len_outer)
