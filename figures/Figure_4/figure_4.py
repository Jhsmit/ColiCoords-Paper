import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from colicoords import Cell, load, save, CellPlot, Data
from mpl_toolkits.axes_grid1 import ImageGrid
from colicoords.support import pad_cell
import os

#cell = load(r'../../datasets/ds3_controls_20170720/lacy_selected_cell_3.cc')

cell = load(r'../../datasets/ds1_c41_epec_lacy\img191c002.cc')
cell_storm = load(r'D:\_processed_data\2018\20180312_yichen_lacy_eyfp\SR\03_14mW\cell4.cc')

save('storm_cell.cc', cell_storm)

data = Data()
for data_elem in cell.data.data_dict.values():
    data.add_data(data_elem, data_elem.dclass, data_elem.name)
cell_raw = Cell(data[:, 1:-1])

storm_padded = pad_cell(cell_storm, (70, 70))

reload = False
if reload:
    cell_storm_opt = storm_padded.copy()
    cell_storm_opt.optimize('storm')
    save('cell_storm_opt.cc', cell_storm_opt)
else:
    cell_bin = load('cell_bin.cc')
    cell_bf = load('cell_bf.cc')
    cell_flu = load('cell_flu.cc')
    cell_storm_opt = load('cell_storm_opt.cc')

fig_width = 8.53534 / 2.54
fig, axes = plt.subplots(2, 2, figsize=(fig_width, (2/2)*fig_width))

for ax in axes.flatten():
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)

upscale = 2

cp = CellPlot(storm_padded)
cp.imshow('binary', ax=axes[0, 0], cmap='gray_r', alpha=0.3)
cp.plot_storm(method='gauss', alpha_cutoff=0.3, ax=axes[0, 0], upscale=upscale, interpolation='spline16')
#cp.plot_storm(method='plot', ax=axes[0, 0], color='b', alpha=0.5, markersize=1)
cp.plot_outline(ax=axes[0, 0], alpha=0.7)
axes[0, 0].plot([15, 15, 27, 27, 15], [28, 40, 40, 28, 28], color='k', linestyle='--')


cp.plot_storm(method='gauss', alpha_cutoff=0.3, ax=axes[0, 1], upscale=upscale, interpolation='spline16')
cp.plot_storm(method='plot', ax=axes[0, 1], color='y', alpha=1, markersize=1.5)
cp.plot_outline(ax=axes[0, 1], alpha=0.7)
cp.plot_midline(ax=axes[0, 1], alpha=0.7)

axes[0, 0].set_xlim(15, 55)
axes[0, 0].set_ylim(55, 15)

axes[0, 1].set_xlim(15, 27)
axes[0, 1].set_ylim(40, 28)

cp = CellPlot(cell_storm_opt)
cp.plot_storm(method='gauss', alpha_cutoff=0.3, ax=axes[1, 0], upscale=upscale, interpolation='spline16')
cp.plot_outline(ax=axes[1, 0], alpha=0.7)
axes[1, 0].plot([15, 15, 27, 27, 15], [28, 40, 40, 28, 28], color='k', linestyle='--')


cp.plot_storm(method='gauss', alpha_cutoff=0.3, ax=axes[1, 1], upscale=upscale, interpolation='spline16')
cp.plot_storm(method='plot', ax=axes[1, 1], color='y', alpha=1, markersize=1.5)
cp.plot_outline(ax=axes[1, 1], alpha=0.7)
cp.plot_midline(ax=axes[1, 1], alpha=0.7)


axes[1, 0].set_xlim(15, 55)
axes[1, 0].set_ylim(55, 15)

axes[1, 1].set_xlim(15, 27)
axes[1, 1].set_ylim(40, 28)

axes[0, 0].set_ylabel("Initial")
axes[1, 0].set_ylabel("Final")
#
# axes[0, 0].set_ylabel('Binary')
# axes[1, 0].set_ylabel('Brightfield')
# axes[2, 0].set_ylabel('Fluorescence')
# axes[3, 0].set_ylabel('STORM')

plt.tight_layout()
#plt.show()
output_folder = r'C:\Users\Smit\MM\Projects\05_Live_cells\manuscripts\ColiCoords\tex\Figures'
plt.savefig(os.path.join(output_folder, 'Figure4.pdf'), bbox_inches='tight')

