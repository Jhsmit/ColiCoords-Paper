import matplotlib.pyplot as plt
import numpy as np
from colicoords import Cell, load, CellPlot
import os
import matplotlib

matplotlib.rcParams.update({'font.size': 8})



cell = load(r'../../data/lacy_selected_cell_3.hdf5')
data = cell.data.copy()
cell_raw = Cell(data)

arrow_kwargs = {'color': 'k', 'head_width': 2, 'head_length': 3, 'length_includes_head': True, 'overhang': 0.2,
                'linewidth': 1}

fig_width = 5.2 # Plos one text width (inches)
fig, axes = plt.subplots(2, 3, figsize=(0.8*fig_width, 2.5))

for ax in axes.flatten():
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)

empty_img = np.zeros_like(cell_raw.data.data_dict['binary'])
empty_img[:] = np.nan

cp = CellPlot(cell_raw)
cp.imshow('binary', ax=axes[0, 0], cmap='gray_r')
cp.plot_midline(ax=axes[0, 0])
cp.plot_outline(ax=axes[0, 0])
axes[0, 0].text(0.05, 0.95, 'A', horizontalalignment='left', verticalalignment='top', transform=axes[0, 0].transAxes, fontsize=15)

ymax, xmax = empty_img.shape
cp.imshow(empty_img, ax=axes[1, 0], cmap='gray_r')
axes[1, 0].axis('off')

axes[1, 0].plot([xmax/2, xmax/2], [0, 0.2*ymax], color='k', linewidth=1)
axes[1, 0].plot([xmax/2, xmax/2], [0.375*ymax, 0.5*ymax], color='k', linewidth=1)
axes[1, 0].arrow(xmax/2, ymax/2, (xmax/2) - 0.05*xmax, 0, **arrow_kwargs)

#axes[1, 0].plot([0, xmax/2, xmax/2], [ymax/2, ymax/2, 0.65*ymax], color='k', linewidth=1)
#axes[1, 0].arrow(xmax/2, 0.85*ymax, 0, 0.15*ymax - 0.05*ymax, **arrow_kwargs)
axes[1, 0].text(0.5, 0.7, 'Calculate $r_c$', horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)

cp.imshow(empty_img, ax=axes[0, 1], cmap='gray_r')
axes[0, 1].axis('off')
axes[0, 1].text(0.5, 0.70, 'Compare and\nupdate parameters', horizontalalignment='center', verticalalignment='center', transform=axes[0, 1].transAxes)
#axes[0, 1].plot([xmax/2, xmax/2], [ymax, 2*ymax/3], color='k', linewidth=1)
axes[0, 1].arrow(xmax, ymax/2, -xmax + 0.05*xmax, 0, **arrow_kwargs)

ymax, xmax = cell_raw.data.shape
r_bk = cell_raw.coords.r
for r in np.arange(5, 40, 5):
    cell_raw.coords.r = r
    cp.plot_outline(ax=axes[1, 1], color='k', linewidth=0.25)
cell_raw.coords.r = r_bk
cp.imshow(cell_raw.coords.rc, ax=axes[1, 1])
axes[1, 1].text(0.05, 0.95, 'B', horizontalalignment='left', verticalalignment='top', transform=axes[1, 1].transAxes,
                fontsize=15)

comparison = (cell_raw.coords.rc < cell_raw.coords.r).astype(int) + cell_raw.data.data_dict['binary']
cp.imshow(comparison, ax=axes[0, 2], cmap='gray_r')
axes[0, 2].text(0.05, 0.95, 'C', horizontalalignment='left', verticalalignment='top', transform=axes[0, 2].transAxes,
                fontsize=15)

cp.imshow(empty_img, ax=axes[1, 2], cmap='gray_r')
axes[1, 2].plot([0, xmax/2], [ymax/2, ymax/2], color='k', linewidth=1)
axes[1, 2].plot([xmax/2, xmax/2], [0.375*ymax, 0.5*ymax], color='k', linewidth=1)

axes[1, 2].arrow(xmax/2, 0.2*ymax, 0, -0.2*ymax + 0.05*ymax, **arrow_kwargs)
axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.7, 'Thresholding', horizontalalignment='center', verticalalignment='center',
                transform=axes[1, 2].transAxes)


plt.tight_layout(w_pad=0, h_pad=0)

#plt.show()
output_folder = r'.'
plt.savefig(os.path.join(output_folder, 'Figure_3.pdf'), bbox_inches='tight', dpi=600)