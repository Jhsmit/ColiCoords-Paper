import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from colicoords import Cell, load, save, CellPlot, Data
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Arc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import AxesImage
from matplotlib import ticker
import os


def calc_dx_dy(cell, x):
    d_xr = cell.coords.p_dx(x)
    dy = 1 / np.sqrt(d_xr ** 2 + 1)
    dx = d_xr / np.sqrt(d_xr ** 2 + 1)
    return dx, dy


cell = load(r'../../data/lacy_selected_cell_3.hdf5')
data = Data()

for data_elem in cell.data.data_dict.values():
    data.add_data(data_elem, data_elem.dclass, data_elem.name)
c = Cell(data)
c.optimize('brightfield')

reconstructed_bf = c.reconstruct_image('brightfield', step=0.5)
c.data.add_data(reconstructed_bf, 'brightfield', 'sim_bf')


cp = CellPlot(c)
fig_width = 8.53534 / 2.54
fig = plt.figure(figsize=(fig_width, 5))
gs0 = GridSpec(1, 1)

ax0 = fig.add_subplot(gs0[0])

cmap = plt.cm.gray
bf = c.data.data_dict['sim_bf']
colors = cmap((bf - bf.min()) / (bf.max() - bf.min()))
colors[..., -1] = np.ones_like(bf)*0.45

cp.imshow(colors)
cp.plot_midline(ax=ax0, color='r', linewidth=1)

vline_kwargs = {'linestyle': '--', 'alpha': 0.75, 'linewidth': 0.5}
ax0.vlines(c.coords.xl, c.coords.p(c.coords.xl), c.data.shape[0], **vline_kwargs)
ax0.vlines(c.coords.xr, c.coords.p(c.coords.xr), 48, **vline_kwargs)
ax0.vlines(c.coords.xr, 58, c.data.shape[0], **vline_kwargs)

#POINT A
xp, yp = 48, 12
xc = c.coords.calc_xc(xp, yp)
xc_A = xc
xp_A, yp_A = xp, yp

yc = c.coords.p(xc)
ax0.plot([xp, xc], [yp, yc], color='k', marker='o', markersize=2)
ax0.vlines(xc, c.coords.p(xc), c.data.shape[0], **vline_kwargs)

dx = xp - xc
dy = yp - yc
ax0.arrow(xp - dx, yp - dy, dx, dy, color='k',
          head_width=2, head_length=3, length_includes_head=True, lw=0, overhang=0.2)


#length coordinate
l_x = np.linspace(c.coords.xl, xc, 100, endpoint=True)
l_y = c.coords.p(l_x) - 2

ax0.plot(l_x[5:-5], l_y[5: -5], color='k')

#arrow
dy, dx = calc_dx_dy(c, l_x[0])
dx *= -5
dy *= -5
ax0.arrow(l_x[0] - dx, l_y[0] - dy, dx, dy, color='k',
          head_width=2, head_length=3, length_includes_head=True, lw=0, overhang=0.2)

dy, dx = calc_dx_dy(c, l_x[-1])
dx *= 5
dy *= 5
ax0.arrow(l_x[-1] - dx, l_y[-1] - dy, dx, dy, color='k',
          head_width=2, head_length=3, length_includes_head=True, lw=0, overhang=0.2)


#POINT B
#xp, yp = 63, 45
xp, yp = 62, 48
xp_B, yp_B = xp, yp
xc = c.coords.xr
yc = c.coords.p(xc)
ax0.plot([xp, xc], [yp, yc], color='k', marker='o', markersize=2)

dx = xp - xc
dy = yp - yc
ax0.arrow(xp - dx, yp - dy, dx, dy, color='k',
          head_width=2, head_length=3, length_includes_head=True, lw=0, overhang=0.2)

#right perpendicular line
rl = 10
d_xr = c.coords.p_dx(c.coords.xr)
dy = rl / np.sqrt(d_xr**2 + 1)
dx = (rl * d_xr) / np.sqrt(d_xr**2 + 1)
ax0.plot([c.coords.xr + dx, c.coords.xr - dx], [c.coords.p(c.coords.xr) - dy, c.coords.p(c.coords.xr) + dy], color='k', linewidth=0.5)

th1 = np.arctan(c.coords.p_dx(c.coords.xr)) * 180 / np.pi

theta2 = c.coords.calc_phi(xp, yp)

arc = Arc((c.coords.xr, c.coords.p(c.coords.xr)), 5, 5, theta1=th1 - 90, theta2=theta2 - 90 + th1, linewidth=0.5)
ax0.add_patch(arc)

#Add text labels
ax0.text(c.coords.xl, 77, '$x_l$', fontsize=14, horizontalalignment='center')
ax0.text(xc_A, 77, '$x_c$', fontsize=14, horizontalalignment='center')
ax0.text(c.coords.xr, 77, '$x_r$', fontsize=14, horizontalalignment='center')

ax0.text(xp_A - 15, yp_A - 4, '$A (x_p, y_p)$', fontsize=10)
ax0.text(xp_B - 17, yp_B + 7, '$B (x_p, y_p)$', fontsize=10)
ax0.text(57, 35, '$\phi$', fontsize=10)
ax0.text(28, 27, '$l_c$', fontsize=10)
ax0.text(47, 25, '$r_c$', fontsize=10)



#Bottom half of the figure
gs = GridSpec(2, 2, top=0.5)
gs_axes = []
ax1 = fig.add_subplot(gs[0, 0])
gs_axes.append(ax1)
ax2 = fig.add_subplot(gs[0, 1])
gs_axes.append(ax2)
ax3 = fig.add_subplot(gs[1, 0])
gs_axes.append(ax3)
ax4 = fig.add_subplot(gs[1, 1])
gs_axes.append(ax4)


for ax in gs_axes + [ax0]:
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(axis='y', labelleft=False)

ax1.set_title('$x_c$')
ax2.set_title('$l_c$')
ax3.set_title('$r_c$')
ax4.set_title('$\phi$')

cp.imshow(c.coords.xc_masked, ax=ax1)
cp.imshow(c.coords.lc, ax=ax2, vmax=40)
cp.imshow(c.coords.rc, ax=ax3, vmin=0, vmax=40)
cp.imshow(c.coords.phi, ax=ax4)

ticks_list = [[20, 40], [0, 20, 40], [0, 10, 20, 30, 40], [0, 60, 120, 180]]

def make_cbar(ax, ticks):
    im = [obj for obj in ax.get_children() if isinstance(obj, AxesImage)][0]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb = fig.colorbar(im, cax=cax, orientation='vertical', ticks=ticks)

for ax, t in zip(gs_axes, ticks_list):
    make_cbar(ax, t)

gs.tight_layout(fig, h_pad=0.01, w_pad=0.01, rect=[None, None, None, 0.6])
gs0.tight_layout(fig, h_pad=0.01, w_pad=0.01, rect=[None, 0.6, None, None])


for ax in gs_axes + [ax0]:
    ax.set_rasterization_zorder(1)

plt.show()
#output_folder = r'C:\Users\Smit\MM\Projects\05_Live_cells\manuscripts\ColiCoords\tex\Figures'
#plt.savefig(os.path.join(output_folder, 'Figure0_test.pdf'), bbox_inches='tight')