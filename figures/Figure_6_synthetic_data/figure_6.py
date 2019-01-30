from colicoords.synthetic_data import SynthCell, SynthCellList, add_readout_noise
from colicoords import load, Data, CellPlot
import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import tifffile
import matplotlib.colors as mcolors


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
cmm = make_colormap(
    [ c('black'), c('magenta')])

cmc = make_colormap(
    [c('black'), c('cyan')])


#https://stackoverflow.com/questions/42475508/how-to-combine-gridspec-with-plt-subplots-to-eliminate-space-between-rows-of-s/42481878
data_dir = r'D:\Projects\CC_paper\figures\Figure_6_synthetic_data'
bf_10000 = tifffile.imread(os.path.join(data_dir, 'temp_dir', 'bf_10000_photons.tif'))
ph = 10000
#names = ['raw', 'brightfield', 'brightfield_DE', 'binary', 'storm_inner']
names = ['raw', 'binary', 'brightfield', 'storm_inner']

c_dict = {name: load(os.path.join(data_dir, 'temp_dir', f'm_cells_ph_{ph}_match_{name}_s100.hdf5')) for name in names}
gt_cells = load(os.path.join(data_dir, 'temp_dir', 'gt_cells_ph_10000_match_raw_s100.hdf5'))

for name in names:
    t_cells = c_dict[name]
    for c1, c2 in zip(gt_cells[:2], t_cells[:2]):
        print(len(c1.data.data_dict['storm_inner']), len(c2.data.data_dict['storm_inner']))
        assert len(c1.data.data_dict['storm_inner']) == len(c2.data.data_dict['storm_inner'])

r_vals_gt = np.concatenate([c.coords.calc_rc(c.data.data_dict['storm_inner']['x'], c.data.data_dict['storm_inner']['y']) for c in gt_cells[:10]])


plt.figure(figsize=(17.8/2.54, 3))

gs1 = gridspec.GridSpec(1, 1)
gs2 = gridspec.GridSpec(4, 1)
gs3 = gridspec.GridSpec(2, 4)

gs1.update(left=0.05, right=0.4, wspace=0)
gs2.update(left=0.4, right=0.55, wspace=0)
gs3.update(left=0.60, right=0.95, wspace=0.05)

ax1 = plt.subplot(gs1[0])
ax1.imshow(bf_10000[0], cmap='gray')
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])
print(gs2)

for name, gs in zip(names, gs2):
    cell = c_dict[name][0]
    cp = CellPlot(cell)
    ax = plt.subplot(gs)
    alpha=1
    if name in ['raw', 'binary']:
        cp.imshow('binary', ax=ax)

    elif name == 'brightfield':
        cp.imshow('brightfield')
    elif name == 'storm_inner':
        cp.imshow(np.zeros(cell.data.shape), cmap='gray')
        cp.plot_storm(data_name=name, ax=ax, method='gauss', upscale=2, cmap=cmc)
        cp.plot_storm(data_name='storm_outer', ax=ax, method='gauss', cmap=cmm, alpha_cutoff=0.25)
        alpha=0.2
    cp.plot_outline(ax=ax, alpha=alpha)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

print(gs3)
print(gs3[0, 2])

for i, name in enumerate(names):
    m_cells = c_dict[name]
    m_vals_gt = np.concatenate(
        [c.coords.calc_rc(c.data.data_dict['storm_inner']['x'], c.data.data_dict['storm_inner']['y']) for c in m_cells[:10]])
    ax = plt.subplot(gs3[0, i])
    ax.plot(m_vals_gt, r_vals_gt)


for gs in gs3:
    ax = plt.subplot(gs)

plt.show()