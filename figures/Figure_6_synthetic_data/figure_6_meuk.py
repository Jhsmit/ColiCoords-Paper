from colicoords.synthetic_data import SynthCell, SynthCellList, add_readout_noise
from colicoords import load, Data, CellPlot
from colicoords.preprocess import filter_binaries, data_to_cells
from colicoords.minimizers import DifferentialEvolution
import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt


from generate_images import generate_images

def gen_im():
    cell_list = load('temp_cells.hdf5')
    # print(type(cell_list[0].data.data_dict['storm_inner']))
    # print(cell_list[0].data.data_dict['storm_inner']['frame'])

    out_dict = generate_images(cell_list, 5, 10, 3, (512, 512))

    np.save('binary.npy', out_dict['binary'])
    np.save('brightfield.npy', out_dict['brightfield'])
    np.save('storm.npy', out_dict['storm_inner'])


def load_im():
    return np.load('binary.npy'), np.load('brightfield.npy'), np.load('storm.npy')


cell_list = load('temp_cells.hdf5')
binary, brightfield, storm = load_im()

plt.imshow(binary[0], extent=[0, 512, 512, 0])
b = storm['frame'] == 1
plt.show()

#
# plt.imshow(binary[0]==1)
# plt.show()
#
# plt.imshow(binary[0])
# plt.show()
#
bin_filter = filter_binaries(binary)

data = Data()
data.add_data(bin_filter.astype(int), 'binary')
data.add_data(brightfield, 'brightfield')
data.add_data(storm, 'storm')
#
print('unique', np.unique(bin_filter))
print('unique', np.unique(binary))

plt.imshow(bin_filter[0], extent=[0, 512, 512, 0])
plt.show()

cells = data_to_cells(data)
c = cells[0]
print(c.name)
#c.optimize('brightfield', minimizer=DifferentialEvolution)
c.optimize()
#c.optimize('storm')
cp = CellPlot(c)
fig, axes = plt.subplots(1, 3)
cp.imshow('binary', ax=axes[0])
cp.plot_outline(ax=axes[0])
cp.imshow('brightfield', ax=axes[1])
cp.plot_storm(ax=axes[1])
cp.plot_outline(ax=axes[2])
cp.plot_storm(ax=axes[2], method='gauss', upscale=2)
cp.show()


r_vals = c.coords.calc_rc(c.data.data_dict['storm']['x'], c.data.data_dict['storm']['y'])
c02_gt = cell_list[0]

cp = CellPlot(c02_gt)
fig, axes = plt.subplots(1, 3)
cp.imshow('binary', ax=axes[0])
cp.plot_outline(ax=axes[0])
cp.imshow('brightfield', ax=axes[1])
cp.plot_storm(ax=axes[1])
cp.plot_outline(ax=axes[2])
cp.plot_storm(ax=axes[2], method='gauss', upscale=2)
cp.show()

r_vals_gt = c02_gt.coords.calc_rc(c02_gt.data.data_dict['storm_inner']['x'], c02_gt.data.data_dict['storm_inner']['y'])

plt.plot(r_vals, r_vals_gt, 'r.')
ax = plt.gca()
ax.set_aspect(1)
# plt.xlim(0.9*r_vals.min(), 1.1*r_vals.max())
# plt.ylim(0.9*r_vals.min(), 1.1*r_vals.max())
# plt.plot([0, 10], [0, 10], 'k')
plt.show()

print('mean diff', np.abs(r_vals - r_vals_gt).mean())

# plt.imshow(out_dict['binary'][0])
# plt.show()
#
# print(type(out_dict['brightfield']))
# bf = out_dict['brightfield'] * 10
# bf_noise = add_readout_noise(bf, 5)
#
# plt.imshow(bf_noise[0], cmap='gray')
# plt.show()
# plt.imshow(bf[0], cmap='gray')
# plt.show()
#
# storm = gen_image_from_storm(out_dict['storm_inner'], (512, 512))
# plt.imshow(storm)
# plt.show()
