import numpy as np
import matplotlib.pyplot as plt
from colicoords import load, CellPlot, save
from colicoords.support import pad_cell
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import seaborn as sns
plt.rcParams['pdf.compression'] = 0
sns.set_style('ticks')
plt.rcParams['image.composite_image'] = False
data_dir = r'.'
if not os.path.exists(os.path.join(data_dir, 'plot_vars')):
    os.mkdir(os.path.join(data_dir, 'plot_vars'))
photons = [500, 1000, 10000]
conditions = ['binary', 'brightfield', 'storm_inner']

labelsize = 7.5
upscale = 15  # STORM render resolution
step = 1  # fraction of points plotted in histogram
linewidth = 0.5

reload = False


#https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
mcm = make_colormap(
    [c('black'), c('magenta')])
ccm = make_colormap(
    [c('black'), c('cyan')])

colors = ['cyan', 'magenta', (0, 0.5, 0.5), (0.5, 0, 0.5)]


def prune(arr):
    arr = arr.flatten()
    arr[np.isinf(arr)] = 10
    return arr[~np.isnan(arr)]


if reload:
    for ph in photons:
        im = np.load(os.path.join(data_dir, 'images', 'bf_noise_{}_photons.npy'.format(ph)))
        np.save(os.path.join(data_dir, 'plot_vars', 'bf_noise_{}_photons.npy'.format(ph)), im[0])

        for condition in conditions:
            cells = load(os.path.join(data_dir, 'cell_obj', 'm_cells_ph_{}_filtered_{}.hdf5'.format(ph, condition)))
            save(os.path.join(data_dir, 'plot_vars', 'cells_{}_{}_photons.hdf5'.format(condition, ph)), cells[:50])


cell_dict = {}
for ph in photons:
    cell_dict[ph] = {}
    for condition in conditions:
        cell_dict[ph][condition] = load(os.path.join(data_dir, 'plot_vars', 'cells_{}_{}_photons.hdf5'.format(condition, ph)))

imgs = {ph: np.load(os.path.join(data_dir, 'plot_vars', 'bf_noise_{}_photons.npy'.format(ph))) for ph in photons}


def make_r_hist(ax, r1, r2, r3, r4, step=1):
    h = ax.hist(r1[::step], bins='fd', linewidth=0.75, histtype='step', color=colors[0])
    h = ax.hist(r1[::step], bins='fd', linewidth=0, alpha=0.2, color=colors[0])

    h = ax.hist(r2[::step], bins='fd', linewidth=0.75, linestyle='-', histtype='step', color=colors[1])
    h = ax.hist(r2[::step], bins='fd', linewidth=0, linestyle='-', alpha=0.2, color=colors[1])

    h = ax.hist(r3[::step], bins='fd', linewidth=1, histtype='step', color=colors[2])

    h = ax.hist(r4[::step], bins='fd', linewidth=1, histtype='step', color=colors[3])

    ax.set_xlim(0.5, 2)


def make_obj_hist(ax, values, step=1):
    binwidth = 0.2
    bins = np.arange(0, 50 + binwidth, binwidth)
    bins_log = 10 ** np.arange(-1, 5.5, 0.05)

    h = ax.hist(values[::step], bins=bins, color='#333333', linewidth=0)
    ax.axvline(1, color='r', linewidth=linewidth, zorder=-1)
    ax.set_xlim(0, 50)

    axins = inset_axes(ax, width='100%', height='100%', bbox_to_anchor=(0.475, 0.4, 0.5, 0.5),
                       bbox_transform=ax.transAxes, loc=1)
    axins.hist(values[::step], bins=bins_log, color='#333333', linewidth=0)
    axins.yaxis.OFFSETTEXTPAD = 1
    axins.set_xscale('log')
    axins.axvline(1, color='r', linewidth=linewidth, zorder=-1)
    axins.tick_params(direction='out', labelsize=labelsize, pad=0)
    axins.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    axins.set_xlim(8e-1, 1000)
    axins.yaxis.offsetText.set_fontsize(labelsize)

    return axins


fig = plt.figure(figsize=(10, 17.8 / 2.54))
outer_grid = gridspec.GridSpec(2, 3, wspace=0.1, hspace=0.15, height_ratios=[0.5, 1])
outer_grid.update(left=0.05, right=0.98, top=0.965)

ci = 0  # Cell index
shapes = np.array([cell_dict[ph]['binary'][ci].data.shape for ph in photons])

max_shape = (shapes.T[0].max(), shapes.T[1].max())


for i, ph in enumerate(photons):
    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 2, width_ratios=[1, 0.45],
                                                  subplot_spec=outer_grid[0, i], wspace=0.0, hspace=0.)

    ax1 = plt.subplot(inner_grid[:, :-1])
    ax1.imshow(imgs[ph], cmap='gray')
    ax1.set_anchor('W')
    ax1.set_title('{} Photons'.format(ph), fontsize=10)

    if ph == 500:
        p0 = ax1.get_position()
        fig.text(0.0, p0.y0 + p0.height, 'A', fontsize=15)

    linewidth = 0.5
    alpha = 0.75

    ax2 = plt.subplot(inner_grid[0, -1])
    ax2.set_anchor('NE')
    ax2.set_title('Binary', fontsize=labelsize)
    cp = CellPlot(pad_cell(cell_dict[ph]['binary'][ci], max_shape))
    cp.imshow("binary", ax=ax2)
    cp.plot_outline(ax=ax2, linewidth=linewidth, alpha=alpha)

    ax3 = plt.subplot(inner_grid[1, -1])
    ax3.set_anchor('E')
    ax3.set_title('Brightfield', fontsize=labelsize)
    cp = CellPlot(pad_cell(cell_dict[ph]['brightfield'][ci], max_shape))
    cp.imshow("brightfield", ax=ax3)
    cp.plot_outline(ax=ax3, linewidth=linewidth, alpha=alpha)

    ax4 = plt.subplot(inner_grid[2, -1])
    ax4.set_anchor('SE')
    ax4.set_title('STORM', fontsize=labelsize)
    cp = CellPlot(pad_cell(cell_dict[ph]['storm_inner'][ci], max_shape))
    cp.imshow(np.zeros(max_shape), cmap='gray', zorder=-2, interpolation='nearest')  # Black background
    cp.plot_outline(ax=ax4, linewidth=linewidth, alpha=1, color='w', zorder=-1)
    cp.plot_storm(data_name='storm_inner', upscale=upscale, method='gauss', alpha_cutoff=0.25, cmap=ccm, interpolation='nearest')
    cp.plot_storm(data_name='storm_outer', upscale=upscale, method='gauss', alpha_cutoff=0.25, cmap=mcm, interpolation='nearest')

    plt.tight_layout(fig)
   # plt.subplots_adjust(left=0, right=1, wspace=0)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks([])
        ax.set_yticks([])

for i, ph in enumerate(photons):
    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 2,
                                                  subplot_spec=outer_grid[1, i], wspace=0.3)

    for j, cond in enumerate(conditions):
        if 1:  # cond == 'binary' and ph == 500:
            # Create r histogram plot
            r_m_inner = prune(np.load(os.path.join(data_dir, 'r_values', 'r_inner_m_ph_{}_{}.npy'.format(ph, cond))))
            r_m_outer = prune(np.load(os.path.join(data_dir, 'r_values', 'r_outer_m_ph_{}_{}.npy'.format(ph, cond))))
            r_gt_inner = prune(np.load(os.path.join(data_dir, 'r_values', 'r_inner_gt_ph_{}_{}.npy'.format(ph, cond))))
            r_gt_outer = prune(np.load(os.path.join(data_dir, 'r_values', 'r_outer_gt_ph_{}_{}.npy'.format(ph, cond))))

            gs = inner_grid[j, 0]
            ax = plt.subplot(gs)
            make_r_hist(ax, r_gt_inner, r_gt_outer, r_m_inner, r_m_outer, step=step)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
            ax.yaxis.offsetText.set_fontsize(labelsize)
            ax.yaxis.offsetText.set_position((-0.15, 1))

            r_m = r_m_inner #np..concatenate((r_m_inner, r_m_outer))
            r_g = r_gt_inner #np.concatenate((r_gt_inner, r_gt_outer))

            d_linear = np.mean(np.abs(r_m - r_g))
            d_sq = np.sqrt(np.mean((r_m - r_g) ** 2))

            ax.text(0.95, 0.85, '$D: {:.2f}$\n$D^2: {:.2f}$'.format(d_linear, d_sq), horizontalalignment='right',
                verticalalignment='center', transform=ax.transAxes, fontsize=labelsize)

            if ph == 500:
                labels = ['Binary', 'Brightfield', 'STORM']
                ax.set_ylabel(labels[j])
                if j == 0:
                    p0 = ax.get_position()
                    fig.text(0.0, p0.y0 + p0.height, 'B', fontsize=15)
            if j == 0:
                ax.set_title('STORM localizations', fontsize=labelsize)
            if j < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Relative distance', fontsize=labelsize)

            ax.tick_params(labelsize=labelsize)

            # Create minimize result histogram
            gs = inner_grid[j, 1]
            ax = plt.subplot(gs)

            obj_vals = np.loadtxt(os.path.join(data_dir, 'obj_values', 'obj_vals_storm_ph_{}_{}.txt'.format(ph, cond)))
            v = obj_vals[:, 0] / obj_vals[:, 1]

            axins = make_obj_hist(ax, v, step=1)

            if j == 0:
                ax.set_title('Optimization result', fontsize=labelsize)
            if j < 2:
                ax.set_xticks([])
            else:
                ax.set_xlabel('Relative chi-squared', fontsize=labelsize)

            ax.tick_params(labelsize=labelsize)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
            ax.yaxis.offsetText.set_fontsize(labelsize)
            ax.yaxis.offsetText.set_position((-0.15, 1))


output_folder = r'.'

plt.savefig(os.path.join(output_folder, 'Figure_6.pdf'), bbox_inches='tight', dpi=1000)

