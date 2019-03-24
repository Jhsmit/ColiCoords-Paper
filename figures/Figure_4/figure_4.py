import matplotlib.pyplot as plt
from colicoords import Cell, load, save, CellPlot, Data
import os

cell = load(r'img191c002.hdf5')

# Copy the cells's data elements to a new data instance and create a new initialized Cell
data = Data()
for data_elem in cell.data.data_dict.values():
    data.add_data(data_elem, data_elem.dclass, data_elem.name)
cell_raw = Cell(data[:, 1:-1])
reload = False

if reload:
    cell_bin = cell_raw.copy()
    cell_bin.optimize()
    save('cell_bin.hdf5', cell_bin)

    cell_bf = cell_raw.copy()
    cell_bf.optimize('brightfield')
    cell_bf.measure_r()
    save('cell_bf.hdf5', cell_bf)

    cell_flu = cell_raw.copy()
    cell_flu.optimize('gain50')
    cell_flu.measure_r()
    save('cell_flu.hdf5', cell_flu)
else:
    cell_bin = load('cell_bin.hdf5')
    cell_bf = load('cell_bf.hdf5')
    cell_flu = load('cell_flu.hdf5')

fig_width = 8.53534 / 2.54
fig, axes = plt.subplots(3, 3, figsize=(fig_width, fig_width))

for ax in axes.flatten():
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)

cp = CellPlot(cell_raw)
cp.imshow('binary', ax=axes[0, 0], cmap='gray_r')
cp.plot_outline(ax=axes[0, 0], alpha=0.5)

cp.imshow(cell_raw.coords.rc < cell_raw.coords.r, ax=axes[0, 1], cmap='gray_r')

cp = CellPlot(cell_bin)
cp.imshow('binary', ax=axes[0, 2], cmap='gray_r')
cp.plot_outline(ax=axes[0, 2], alpha=0.5)

cp = CellPlot(cell_raw)
cp.imshow('brightfield', ax=axes[1, 0], cmap='gray')
cp.plot_outline(ax=axes[1, 0], alpha=0.5)

# Reconstruct the image from the coordinate system
cp.imshow(cell_raw.reconstruct_image('brightfield'), ax=axes[1, 1], cmap='gray')

cp = CellPlot(cell_bf)
cp.imshow('brightfield', ax=axes[1, 2], cmap='gray')
cp.plot_outline(ax=axes[1, 2], alpha=0.5)

cp = CellPlot(cell_raw)
cp.imshow('gain50', ax=axes[2, 0])
cp.plot_outline(ax=axes[2, 0], alpha=0.5)

# Reconstruct the image from the coordinate system
cp.imshow(cell_raw.reconstruct_image('gain50'), ax=axes[2, 1])

cp = CellPlot(cell_flu)
cp.imshow('gain50', ax=axes[2, 2])
cp.plot_outline(ax=axes[2, 2], alpha=0.5)

axes[0, 0].set_title("A")
axes[0, 1].set_title("A")
axes[0, 2].set_title("A")

axes[0, 0].set_ylabel('A')
axes[1, 0].set_ylabel('A')
axes[2, 0].set_ylabel('A')

plt.tight_layout()

axes[0, 0].set_title("Initial")
axes[0, 1].set_title("Calculated")
axes[0, 2].set_title("Final")

axes[0, 0].set_ylabel('Binary')
axes[1, 0].set_ylabel('Brightfield')
axes[2, 0].set_ylabel('Fluorescence')

output_folder = r'.'
plt.savefig(os.path.join(output_folder, 'Figure_4.pdf'), bbox_inches='tight', dpi=1000)

