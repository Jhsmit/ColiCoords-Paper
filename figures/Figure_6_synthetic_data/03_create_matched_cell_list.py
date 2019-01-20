import numpy as np
from colicoords import load, save, Data, filter_binaries, data_to_cells, CellPlot, CellListPlot, CellList
from colicoords.minimizers import DifferentialEvolution
import tifffile
import mahotas as mh
import re
from tqdm.auto import tqdm
import fastcluster as fc
from scipy.cluster.hierarchy import fcluster


def encode(arr):
    encode_arr = (200 * np.arange(len(arr)) + 1) ** 2
    return np.sum(arr * encode_arr)


def encode_intensity(cells):
    out_arr = np.empty(len(cells), dtype=int)
    for i, cell in enumerate(cells):
        intensity = cell.data.data_dict['storm_inner']['intensity']
        out_arr[i] = encode(intensity)

    return out_arr


def match_cells(gt_cells, m_cells, storm_input, filtered_binaries, max_d=3):
    img_numbers = np.array([int(re.findall(r'(\d+)', cell.name)[0]) for cell in m_cells])
    encoded_gt = encode_intensity(gt_cells)

    false_positive = 0
    bordering = 0
    too_far = 0
    max_idx_gt = 0

    gt_matched, m_matched = [], []
    for i in tqdm(np.unique(storm_input['frame'])):  # Iteration starts at 1 (ImageJ indexing)
        st_elem = storm_input[storm_input['frame'] == i].copy()
        X = np.array([st_elem['x'], st_elem['y']]).T.copy()
        linkage = fc.linkage(X)
        clusters = fcluster(linkage, max_d, criterion='distance')
        clustered_st = [st_elem[clusters == i] for i in np.unique(clusters)]
        encoded_storm = [encode(elem['intensity']) for elem in clustered_st]

        s_cells = m_cells[img_numbers == (i - 1)]
        if len(s_cells) == 0:
            print('No cells, img {}'.format(i))
            continue

        cell_numbers = np.array([int(re.findall(r'(\d+)', cell.name)[1]) for cell in s_cells])
        binary_img = filtered_binaries[i - 1]
        coms_cells = np.array([mh.center_of_mass(binary_img == j) for j in cell_numbers])

        matched = 0
        for cluster, code in zip(clustered_st, encoded_storm):

            # Find the GT cell
            idx_gt = np.argwhere(code == encoded_gt)
            if idx_gt > max_idx_gt:
                max_idx_gt = idx_gt

            if len(idx_gt) == 0:
                # print('Cluster not in cells, probably bordering cell')
                bordering += 1
                continue
            else:
                gt_cell = gt_cells[idx_gt[0][0]]

            # Find the M cell
            com_storm = [np.mean(cluster['y']), np.mean(cluster['x'])]
            ds = np.sqrt((coms_cells[:, 0] - com_storm[0]) ** 2 + (coms_cells[:, 1] - com_storm[1]) ** 2)

            idx_m = np.argmin(ds)
            if np.min(ds) > 20:
                too_far += 1
                continue
            else:
                matched += 1
                m_cell = s_cells[idx_m]

            gt_matched.append(gt_cell.name + '\n')
            m_matched.append(m_cell.name + '\n')

        false_positive += (len(s_cells) - matched)

    print('False positive', false_positive)
    print('Bordering, Too far', bordering, too_far)
    print('Max GT index:', max_idx_gt)

    return gt_matched, m_matched


def match_all():
    print('Loading GT')
    gt_cells = load('cells_final_selected.hdf5')
    storm_i = np.load('storm_inner.npy')

    for ph in [10000, 1000, 500]:
        print(f'Photons {ph}')

        m_cells = load('cell_obj/cell_ph_{}_raw.hdf5'.format(ph))
        print('Measured cells loaded')

        bin_predicted = tifffile.imread('binary_{}photons_predicted.tif'.format(ph))
        print('Filtering')
        filtered_pred = filter_binaries(bin_predicted, min_size=495, max_size=2006.4, min_minor=7.57, max_minor=17.3,
                                        min_major=15.41, max_major=54.97)

        gt_match, m_match, = match_cells(gt_cells, m_cells, storm_i, filtered_pred, max_d=5)

        print('Matched {} cells out of max {}'.format(len(m_match), len(m_cells)))

        for i, (m_, gt_) in tqdm(enumerate(zip(m_match, gt_match))):
            m_i = m_cells.name.tolist().index(m_.rstrip())
            g_i = gt_cells.name.tolist().index(gt_.rstrip())

            try:
                assert len(m_cells[m_i].data.data_dict['storm_inner']) == len(gt_cells[g_i].data.data_dict['storm_inner'])
            except AssertionError:
                print('Assertion error:', i)

        with open('matched_names/gt_cells_ph_{}_match.txt'.format(ph), 'w') as f:
            f.writelines(gt_match)

        with open('matched_names/m_cells_ph_{}_match.txt'.format(ph), 'w') as f:
            f.writelines(m_match)



def gen_cells():
    storm_i = np.load('storm_inner.npy')
    storm_o = np.load('storm_outer.npy')
    foci_i = np.load('foci_inner.npy')
    foci_o = np.load('foci_outer.npy')

    for ph in [500, 1000, 10000]:
        print(f'Photons {ph}')
        bin_predicted = tifffile.imread('binary_{}photons_predicted.tif'.format(ph))
        bf = np.load('bf_noise_{}_photons.npy'.format(ph))

        print('Filtering')
        filtered_pred = filter_binaries(bin_predicted, min_size=495, max_size=2006.4, min_minor=7.57, max_minor=17.3,
                                        min_major=15.41, max_major=54.97)

        data = Data()
        data.add_data(filtered_pred, 'binary')
        data.add_data(bf, 'brightfield')
        data.add_data(storm_i, 'storm', 'storm_inner')
        data.add_data(storm_o, 'storm', 'storm_outer')
        data.add_data(foci_i, 'fluorescence', 'foci_inner')
        data.add_data(foci_o, 'fluorescence', 'foci_outer')

        print('Making cells')
        m_cells = data_to_cells(data, remove_multiple_cells=False, remove_bordering=False)

        print('Saving')
        save('cell_obj/cell_ph_{}_raw.hdf5'.format(ph), m_cells)


def optimize_all():
    # for ph in [500]:
    #     print(f'Photons {ph}')
    #
    #     print('Measured cells loaded')
    #
    #     print('binary')
    #     optimize_cells = m_cells.copy()
    #     optimize_cells.optimize_mp()
    #     save('cell_obj/m_cells_ph_{}_match_binary.hdf5'.format(ph), optimize_cells)
    #
    #     print('brightfield')
    #     optimize_cells = m_cells.copy()
    #     optimize_cells.optimize_mp('brightfield')
    #     save('cell_obj/m_cells_ph_{}_match_brightfield.hdf5'.format(ph), optimize_cells)
    #
    #     print('storm inner')
    #     optimize_cells = m_cells.copy()
    #     optimize_cells.optimize_mp('storm_inner')
    #     save('cell_obj/m_cells_ph_{}_match_storm_inner.hdf5'.format(ph), optimize_cells)

    for ph in [10000, 1000, 500]:
        print(f'Photons {ph}')
        m_cells = load('cell_obj/m_cells_ph_{}_match_raw.hdf5'.format(ph))

        print('brightfield DE')
        optimize_cells = m_cells.copy()
        optimize_cells.optimize_mp('brightfield', minimizer=DifferentialEvolution)
        save('cell_obj/m_cells_ph_{}_match_brightfield_DE.hdf5'.format(ph), optimize_cells)


if __name__ == '__main__':
    match_all()

    #
    # binary  = np.load('binary.npy')
    #
    # plt.imshow(binary[206])
    # plt.show()
    # numbers = np.load('img_number.npy')
    #
    # plt.hist(numbers)
    # plt.show()
    #
    # print(np.unique(numbers))
    # print(206 in numbers)
    # match_all()

    #gt_cells = load('cells_final_selected.hdf5')
    #st = np.array([c.data.data_dict['storm_outer']['intensity'] for c in gt_cells])
    #np.save('storm_int.npy', st)

    # #img_number = np.array([int(re.findall(r'(\d+)', c.name)[0]) for c in gt_cells])
    # #np.save('img_number.npy', img_number)
    # img_number = np.load('img_number.npy')
    # st = np.load('storm_int.npy')

    #encoded = encode_intensity(gt_cells)
    #np.save('encoded.npy', encoded)
    #encoded = np.load('encoded.npy')
    #cells = load('cell_obj/cell_ph_10000_raw.hdf5')
    # c1 = load('cell_obj/gt_cells_ph_500_match_raw.hdf5')
    # c2 = load('cell_obj/m_cells_ph_500_match_raw.hdf5')
    # print(c1[0].data.names)
    # print([c1[0].data.data_dict[name].dtype for name in c1[0].data.names])
    # print([c1[0].data.data_dict[name].shape for name in c1[0].data.names])
    # print(c2[0].data.names)
    # print([c2[0].data.data_dict[name].dtype for name in c2[0].data.names])
    # print([c2[0].data.data_dict[name].shape for name in c2[0].data.names])
    #cell = cells[75]
    #save('cell75.h5', cell)
    #cell = load('cells.hdf5')

    # bin_predicted = tifffile.imread('binary_{}photons_predicted.tif'.format(10000))
    # filtered_pred = filter_binaries(bin_predicted, min_size=495, max_size=2006.4, min_minor=7.57, max_minor=17.3,
    #                                 min_major=15.41, max_major=54.97)
    #
    # np.save('filtered.npy', filtered_pred)
    # binary = np.load('binary.npy')
    #
    # img_n, c_n = (int(n) for n in re.findall(r'(\d+)', c.name))
    # print(img_n, c_n)
    #
    # orig_n = np.int(np.unique(binary[img_n][filtered_pred[img_n] == c_n])[-1])
    # print(orig_n)


    # print(np.unique(binary[1]))
    # nums = [int(np.max(a)) for a in binary]
    # cells_chunked = list(chunk_list(cells, nums))
    #matched_cells.append(cells_chunked[img_n][orig_n - 1])


    # cells = load('cell_obj/m_cells_ph_10000_fail_raw.hdf5')
    # c = cells[0]
    # c_n = int(re.findall(r'(\d+)', c.name)[0])
    # print(c_n)
    # b = img_number == c_n
    # st_c = c.data.data_dict['storm_outer']['intensity']
    # print('search')
    # sums = [np.sum([elem in st_elem for elem in st_c]) for st_elem in st[b]]
    # print(np.argmax(sums))
    # print(np.max(sums))


    cells = load('cell_obj/m_cells_ph_500_fail1_raw.hdf5')
  #
    cell = cells[-1]
    cp = CellPlot(cell)
    cp.imshow('binary')
    cp.plot_storm(data_name='storm_outer', color='g')
    cp.plot_storm(data_name='storm_inner', color='b')
    cp.plot_midline()
    cp.plot_outline()
    cp.show()
  #
  #   # cp.hist_r_storm(data_name='storm_outer', norm_x=False)
  #   # cp.show()
  #
  #   #cell = cells[6]
  # #  save('cells.hdf5', cell)
  #
  #   r_max = 15
  #
  #   r = cell.coords.calc_rc(cell.data.data_dict['storm_outer']['x'], cell.data.data_dict['storm_outer']['y'])
  #   b = r < r_max
  #
  #
  #   print(np.sum([~b]))
  #
  #   r = cell.coords.calc_rc(cell.data.data_dict['storm_inner']['x'], cell.data.data_dict['storm_inner']['y'])
  #   b = r < r_max
  #
  #   xs = np.round(cell.data.data_dict['storm_inner']['x']).astype(int)
  #   ys = np.round(cell.data.data_dict['storm_inner']['y']).astype(int)
  #
  #   plt.figure()
  #   plt.imshow(cell.data.binary_img)
  #   plt.show()
  #
  #   print(np.unique(cell.data.binary_img))
  #
  #   pixels = cell.data.binary_img[ys, xs]
  #   print(pixels.shape)
  #   print(np.unique(pixels))
  #
  #   print('all true', np.all(pixels))
  #
  #   print(cell.data.binary_img.shape)
  #   print(cell.data.binary_img.dtype)
  #
  #   print(np.sum([~pixels]))
  #   val = encode_intensity([cell])
  #
  #   print(val)
  #   print(val[0] in encoded)
  #   print(np.min(np.abs(val[0] - encoded)))
  #
  #   #
  #   #
  #   #
  #   # binary, bf, storm_i, storm_o, bin_pred = load_tempfiles()
  #   # bin_predicted = tifffile.imread('binary_500photons_predicted.tif')
  #   # bin_pred = bin_predicted[0:binary.shape[0]]
  #   #
  #   # bf = np.load('bf_noise_500_photons.npy')[0:binary.shape[0]]
  #   #
  #   # labeled_pred = np.zeros_like(bin_pred, dtype=int)
  #   # for i, img in enumerate(bin_pred):
  #   #     labeled_pred[i], n = mh.labeled.label(img)
  #   # filtered_pred = filter_binaries(labeled_pred, min_size=495, max_size=2006.4, min_minor=7.57, max_minor=17.3,
  #   #                                 min_major=15.41, max_major=54.97)
  #   #
  #   #
  #   # #
  #   # # print(np.unique(binary[1]))
  #   # # nums = [int(np.max(a)) for a in binary]
  #   # # cells_chunked = list(chunk_list(cells, nums))
  #   # # print(nums)
  #   #
  #   # # fig, axes = plt.subplots(1, 2)
  #   # # axes[0].imshow(labeled_pred[0])
  #   # # axes[1].imshow(filtered_pred[0])
  #   # # plt.show()
  #   # #
  #   # fig, axes = plt.subplots(1, 2)
  #   # axes[0].imshow(binary[0])
  #   # axes[1].imshow(filtered_pred[0])
  #   # plt.show()
  #   #
  #   # data = Data()
  #   # data.add_data(filtered_pred, 'binary')
  #   # data.add_data(bf, 'brightfield')
  #   # data.add_data(storm_i, 'storm', 'storm_inner')
  #   # data.add_data(storm_o, 'storm', 'storm_outer')
  #   # m_cells = data_to_cells(data, remove_multiple_cells=False, remove_bordering=False)
  #   #
  #   # gt_matched, m_matched = match_cells(gt_cells, m_cells)
  #   #
  #   #
  #   #
  #   #
  #   # # matched_cells = []
  #   # # for c in cells_r[:20]:
  #   # #     img_n, c_n = (int(n) for n in re.findall(r'(\d+)', c.name))
  #   # #     print(img_n, c_n)
  #   # #
  #   # #     orig_n = np.int(np.unique(binary[img_n][filtered_pred[img_n] == c_n])[-1])
  #   # #     print(orig_n)
  #   # #
  #   # #     matched_cells.append(cells_chunked[img_n][orig_n - 1])
