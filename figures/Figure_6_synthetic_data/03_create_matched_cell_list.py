import numpy as np
from colicoords import load, save, Data, filter_binaries, data_to_cells
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
    """For each ground-truth cell find the corresponding 'measured' cell"""
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


def gen_cells():
    storm_i = np.load('images/storm_inner.npy')
    storm_o = np.load('images/storm_outer.npy')
    foci_i = np.load('images/foci_inner.npy')
    foci_o = np.load('images/foci_outer.npy')

    for ph in [10000, 1000, 500]:
        print('Photons {}'.format(ph))
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


def match_all():
    """For all conditions match all ground-truth cells to measured cells"""
    print('Loading GT')
    gt_cells = load('cells_final_selected.hdf5')
    storm_i = np.load('storm_inner.npy')

    for ph in [10000, 1000, 500]:
        print('Photons {}'.format(ph))

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


def optimize_all():
    """Optimize the cell's coordinate systems for each conditions based on different data elements"""
    for ph in [10000, 1000, 500]:
        print('Photons {}'.format(ph))
        m_cells = load('cell_obj/m_cells_ph_{}_match_raw.hdf5'.format(ph))

        print('Measured cells loaded')

        print('binary')
        optimize_cells = m_cells.copy()
        optimize_cells.optimize_mp()
        save('cell_obj/m_cells_ph_{}_match_binary.hdf5'.format(ph), optimize_cells)

        print('brightfield')
        optimize_cells = m_cells.copy()
        optimize_cells.optimize_mp('brightfield')
        save('cell_obj/m_cells_ph_{}_match_brightfield.hdf5'.format(ph), optimize_cells)

        print('storm inner')
        optimize_cells = m_cells.copy()
        optimize_cells.optimize_mp('storm_inner')
        save('cell_obj/m_cells_ph_{}_match_storm_inner.hdf5'.format(ph), optimize_cells)


if __name__ == '__main__':
    gen_cells()
    match_all()
    optimize_all()
