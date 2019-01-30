from colicoords import load, save
from colicoords.postprocess import align_data_element
import matplotlib.pyplot as plt

pth = r'D:\_processed_data\2018\20181010_eYFP_EscV'
cells_measured = load(r'D:\_processed_data\2018\20181010_eYFP_EscV\cells_r_selected.hdf5')
cells_synth = load('temp_cells.hdf5')
print(type(cells_measured[0]))
print(type(cells_measured[1:2]))
print(len(cells_measured[1:2]))
out = align_data_element(cells_synth[0], cells_measured[100:101], 'brightfield', r_norm=True)

plt.imshow(out)
plt.show()

