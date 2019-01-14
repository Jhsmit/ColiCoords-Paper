import numpy as np
from colicoords import load, save, Data, filter_binaries, data_to_cells, CellPlot, CellListPlot
import matplotlib.pyplot as plt
import tifffile
import mahotas as mh
import re



def gen_tempfiles():
    binary, bf, storm =  np.load('binary.npy'), np.load('brightfield.npy'), np.load('storm.npy')
    bin_predicted = tifffile.imread('binary_10000photons_predicted.tif')
    np.save('temp/binary.npy', binary[:20])
    np.save('temp/brightfield.npy', bf[:20])
    np.save('temp/storm.npy', storm)
    np.save('temp/bf_pred.npy', bin_predicted[:20])


def load_tempfiles():

    binary = np.load('temp/binary.npy')
    bf = np.load('temp/brightfield.npy')
    storm = np.load('storm_inner.npy')
    bin_pred = np.load('temp/bf_pred.npy')

    return binary, bf, storm, bin_pred


def chunk_list(l, sizes):
    prev = 0
    for s in sizes:
        result = l[prev:prev+s]
        prev += s
        yield result


if __name__ == '__main__':
    cells = load('temp/cells_temp.hdf5')

    cp = CellPlot(cells[12])
    cp.imshow('binary')
    cp.show()

    binary, bf, storm, bin_pred = load_tempfiles()
    labeled_pred = np.zeros_like(bin_pred, dtype=int)
    for i, img in enumerate(bin_pred):
        labeled_pred[i], n = mh.labeled.label(img)
    filtered_pred = filter_binaries(labeled_pred)

    print(np.unique(binary[1]))
    nums = [int(np.max(a)) for a in binary]
    cells_chunked = list(chunk_list(cells, nums))
    print(nums)

    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(labeled_pred[0])
    # axes[1].imshow(filtered_pred[0])
    # plt.show()
    #
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(binary[0])
    axes[1].imshow(filtered_pred[0])
    plt.show()

    data = Data()
    data.add_data(filtered_pred, 'binary')
    data.add_data(bf, 'brightfield')
    data.add_data(storm, 'storm')
    cells_r = data_to_cells(data, remove_multiple_cells=False, remove_bordering=False)

    print(len(cells_r))


    matched_cells = []
    for c in cells_r[:20]:
        img_n, c_n = (int(n) for n in re.findall(r'(\d+)', c.name))
        print(img_n, c_n)

        orig_n = np.int(np.unique(binary[img_n][filtered_pred[img_n] == c_n])[-1])
        print(orig_n)

        matched_cells.append(cells_chunked[img_n][orig_n - 1])

    i = 0
    c_r = cells_r[i]
    c_m = matched_cells[i]
    cp_r = CellPlot(cells_r[i])
    cp_m = CellPlot(matched_cells[i])
    c_r.optimize('storm')

    fig, axes = plt.subplots(1, 2)
    cp_r.imshow('brightfield', ax=axes[0])
    cp_r.plot_outline(ax=axes[0])
    cp_r.plot_storm(data_name='storm', ax=axes[0], color='g')
    cp_m.imshow('brightfield', ax=axes[1])
    cp_m.plot_storm(data_name='storm_inner', ax=axes[1], color='g')
    cp_m.plot_outline(ax=axes[1])
    plt.show()


    r_vals = c_r.coords.calc_rc(c_r.data.data_dict['storm']['x'], c_r.data.data_dict['storm']['y'])
    r_vals_gt = c_m.coords.calc_rc(c_m.data.data_dict['storm_inner']['x'], c_m.data.data_dict['storm_inner']['y'])

    plt.figure()
    plt.plot(r_vals, r_vals_gt, 'r.')
    plt.show()