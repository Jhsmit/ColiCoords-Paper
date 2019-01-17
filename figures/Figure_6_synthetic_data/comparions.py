import numpy as np
from colicoords import load, save, Data, filter_binaries, data_to_cells, CellPlot, CellListPlot
import matplotlib.pyplot as plt
import tifffile
import mahotas as mh
import re



def encode_intensity(cells, r_max=9):
    out_arr = np.empty(len(cells))
    for i, cell in enumerate(cells):
        int_outer = cell.data.data_dict['storm_outer']['intensity']
        r = cell.coords.calc_rc(cell.data.data_dict['storm_outer']['x'], cell.data.data_dict['storm_outer']['y'])
        b = r < r_max
        encode_arr = (200*np.arange(len(int_outer)) + 1) ** 2
        s1 = np.sum(int_outer[b] * encode_arr[b])

        int_inner = cell.data.data_dict['storm_inner']['intensity']
        r = cell.coords.calc_rc(cell.data.data_dict['storm_inner']['x'], cell.data.data_dict['storm_inner']['y'])
        b = r < r_max
        encode_arr = (200*np.arange(len(int_inner)) + 1) ** 2
        s2 = np.sum(int_inner[b] * encode_arr[b])

        out_arr[i] = s1 + s2

    return out_arr


def match_cells(gt_cells, m_cells):
        """Iterate though measured cells m_cells and tries to find corresponding cell in ground-truth cells"""
        gt_out, m_out = [], []

        storm_int_gt = encode_intensity(gt_cells)
        assert len(storm_int_gt) == len(np.unique(storm_int_gt))

        storm_int_m = encode_intensity(m_cells)
        zeros = storm_int_m == 0  # cells with zero intensity are false positive
        assert len(storm_int_m[~zeros]) == len(np.unique(storm_int_m[~zeros]))
        print('bools total', np.sum([~zeros]))

        n = 0
        for k, (cell, int_val) in enumerate(zip(m_cells[~zeros], storm_int_m[~zeros])):
            try:
                i = list(storm_int_gt).index(int_val)
                gt_out.append(gt_cells[i])
                m_out.append(cell)
            except ValueError:
                print(k)
                n += 1
                continue

        print('Identified {} cells, not found {} out of total {}'.format(len(gt_out), n, len(m_cells)))

        return gt_out, m_out


def match_all():
    print('Loading GT')
    gt_cells = load('cells_final_selected.hdf5')
    for ph in [10000, 1000, 500]:
        print(f'photons {ph}')

        m_cells = load('cell_obj/cell_ph_{}_raw.hdf5'.format(ph))
        gt_match, m_match = match_cells(gt_cells, m_cells)
        save('cell_obj/gt_cells_ph_{}_match_raw.hdf5'.format(ph), CellList(gt_match))
        save('cell_obj/m_cells_ph_{}_match_raw.hdf5'.format(ph), CellList(m_match))


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
    storm_i = np.load('storm_inner.npy')
    storm_o = np.load('storm_outer.npy')
    bin_pred = np.load('temp/bf_pred.npy')

    return binary, bf, storm_i, storm_o, bin_pred


def chunk_list(l, sizes):
    prev = 0
    for s in sizes:
        result = l[prev:prev+s]
        prev += s
        yield result


def match_cells(gt_cells, m_cells):
        """Iterate though measured cells m_cells and tries to find corresponding cell in ground-truth cells"""
        gt_out, m_out = [], []

        storm_int_gt = [c.data.data_dict['storm_inner']['intensity'].sum() +
                        c.data.data_dict['storm_outer']['intensity'].sum() for c in gt_cells]
        assert len(storm_int_gt) == len(np.unique(storm_int_gt))

        storm_int_m = [c.data.data_dict['storm_inner']['intensity'].sum() +
                        c.data.data_dict['storm_outer']['intensity'].sum() for c in m_cells]
        assert len(storm_int_m) == len(np.unique(storm_int_m))

        n = 0
        for cell, int_val in zip(m_cells, storm_int_m):
            try:
                i = storm_int_gt.index(int_val)
                gt_out.append(gt_cells[i])
                m_out.append(cell)
            except ValueError:
                n += 1
                continue

        print('Identified {} cells, not found {} out of total {}'.format(len(gt_out), n, len(m_cells)))

        return gt_out, m_out


if __name__ == '__main__':
    gt_cells = load('temp/cells_temp.hdf5')

    binary, bf, storm_i, storm_o, bin_pred = load_tempfiles()
    bin_predicted = tifffile.imread('binary_500photons_predicted.tif')
    bin_pred = bin_predicted[0:binary.shape[0]]

    bf = np.load('bf_noise_500_photons.npy')[0:binary.shape[0]]

#    labeled_pred = np.zeros_like(bin_pred, dtype=int)
    # for i, img in enumerate(bin_pred):
    #     labeled_pred[i], n = mh.labeled.label(img)
    filtered_pred = filter_binaries(bin_pred, min_size=495, max_size=2006.4,
                                    min_minor=7.57, max_minor=17.3, min_major=15.41, max_major=54.97)


    #
    # print(np.unique(binary[1]))
    # nums = [int(np.max(a)) for a in binary]
    # cells_chunked = list(chunk_list(cells, nums))
    # print(nums)

    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(labeled_pred[0])
    # axes[1].imshow(filtered_pred[0])
    # plt.show()
    #
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(binary[0])
    axes[1].imshow(filtered_pred[0])
    axes[2].imshow(bf[0])
    plt.show()

    data = Data()
    data.add_data(filtered_pred, 'binary')
    data.add_data(bf, 'brightfield')
    data.add_data(storm_i, 'storm', 'storm_inner')
    data.add_data(storm_o, 'storm', 'storm_outer')
    m_cells = data_to_cells(data, remove_multiple_cells=False, remove_bordering=False)

    gt_matched, m_matched = match_cells(gt_cells, m_cells)




    # matched_cells = []
    # for c in cells_r[:20]:
    #     img_n, c_n = (int(n) for n in re.findall(r'(\d+)', c.name))
    #     print(img_n, c_n)
    #
    #     orig_n = np.int(np.unique(binary[img_n][filtered_pred[img_n] == c_n])[-1])
    #     print(orig_n)
    #
    #     matched_cells.append(cells_chunked[img_n][orig_n - 1])
    #
    i = 0
    gt_c = gt_matched[i]
    m_c = m_matched[i]
    gt_cp = CellPlot(gt_c)
    m_cp = CellPlot(m_c)

    m_c.optimize('brightfield')

    fig, axes = plt.subplots(1, 2)
    gt_cp.imshow('brightfield', ax=axes[0])
    gt_cp.plot_outline(ax=axes[0])
    gt_cp.plot_storm(data_name='storm_inner', ax=axes[0], color='g')
    m_cp.imshow('brightfield', ax=axes[1])
    m_cp.plot_storm(data_name='storm_inner', ax=axes[1], color='g')
    m_cp.plot_outline(ax=axes[1])
    plt.show()
    #
    #
    r_vals = gt_c.coords.calc_rc(gt_c.data.data_dict['storm_inner']['x'], gt_c.data.data_dict['storm_inner']['y'])
    r_vals_gt = m_c.coords.calc_rc(m_c.data.data_dict['storm_inner']['x'], m_c.data.data_dict['storm_inner']['y'])

    plt.figure()
    plt.plot(r_vals, r_vals_gt, 'r.')
    plt.show()

    plt.figure()
    gt_cp.hist_r_storm(data_name='storm_inner', histtype='step', label='gt_inner', norm_x=False)
    gt_cp.hist_r_storm(data_name='storm_outer', histtype='step', label='gt_outer', norm_x=False)

    m_cp.hist_r_storm(data_name='storm_outer', histtype='step', label='m_outer', norm_x=False)
    m_cp.hist_r_storm(data_name='storm_inner', histtype='step', label='m_inner', norm_x=False)
    #plt.gca().set_xlim(0, 0.9)
    plt.legend()

    plt.show()
