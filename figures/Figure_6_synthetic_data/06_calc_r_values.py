import numpy as np
from colicoords import load
import os


def get_r_vals(cell_obj):
    """Get radial distances for inner and outer membranes for the cell object"""
    r_i = cell_obj.coords.calc_rc(cell_obj.data.data_dict['storm_inner']['x'],
                                  cell_obj.data.data_dict['storm_inner']['y'])
    r_o = cell_obj.coords.calc_rc(cell_obj.data.data_dict['storm_outer']['x'],
                                  cell_obj.data.data_dict['storm_outer']['y'])

    return r_i, r_o


def process_cells(m_cell, gt_cell):
    """Measured cell and ground truth cell and returns r values for inner and outer membrane for measured and
    ground truth, respectively"""
    rc = gt_cell.coords.r / (1.5554007217841803 * 1.314602664567288)
    r1 = get_r_vals(m_cell)
    r2 = get_r_vals(gt_cell)

    return list(r / rc for r in r1 + r2)


def get_r_vals_all(data_dir):
    gt_cells = load(os.path.join(data_dir, 'cell_obj', 'cells_final_selected.hdf5'))

    for ph in [10000, 1000, 500]:
        print('Photons', ph)
        m_names = np.genfromtxt(os.path.join(data_dir, 'matched_names', 'm_cells_ph_{}_match_filter.txt'.format(ph)), dtype=str)
        gt_names = np.genfromtxt(os.path.join(data_dir, 'matched_names', 'gt_cells_ph_{}_match_filter.txt'.format(ph)), dtype=str)

        for condition in ['binary', 'brightfield', 'storm_inner']:
            print('Condition', condition)
            m_cells = load(os.path.join(data_dir, 'cell_obj', 'm_cells_ph_{}_filtered_{}.hdf5'.format(ph, condition)))

            # Get index arrays to sort saved cell lists by matched names.
            m_index = np.searchsorted(m_cells.name, m_names)
            gt_index = np.searchsorted(gt_cells.name, gt_names)

            # sorting CellList object by indexing; no copying is done.
            m_sorted = m_cells[m_index]
            gt_sorted = gt_cells[gt_index]

            result = np.array([process_cells(m, gt) for m, gt in zip(m_sorted, gt_sorted)])

            out_arr = np.full((len(m_sorted), 4, 720), fill_value=np.nan)  # Max number of localizations per cell < 720
            for (r0, r1, r2, r3), elem in zip(result, out_arr):  # m_inner, m_outer, gt_inner, gt_outer
                elem[0][:len(r0)] = r0
                elem[1][:len(r1)] = r1
                elem[2][:len(r2)] = r2
                elem[3][:len(r3)] = r3

            np.savetxt(os.path.join(data_dir, 'r_values', 'r_inner_m_ph_{}_{}.txt'.format(ph, condition)),
                       out_arr[:, 0, :])
            np.savetxt(os.path.join(data_dir, 'r_values', 'r_outer_m_ph_{}_{}.txt'.format(ph, condition)),
                       out_arr[:, 1, :])
            np.savetxt(os.path.join(data_dir, 'r_values', 'r_inner_gt_ph_{}_{}.txt'.format(ph, condition)),
                       out_arr[:, 2, :])
            np.savetxt(os.path.join(data_dir, 'r_values', 'r_outer_gt_ph_{}_{}.txt'.format(ph, condition)),
                       out_arr[:, 3, :])

            np.save(os.path.join(data_dir, 'r_values', 'r_inner_m_ph_{}_{}.npy'.format(ph, condition)),
                    out_arr[:, 0, :])
            np.save(os.path.join(data_dir, 'r_values', 'r_outer_m_ph_{}_{}.npy'.format(ph, condition)),
                    out_arr[:, 1, :])
            np.save(os.path.join(data_dir, 'r_values', 'r_inner_gt_ph_{}_{}.npy'.format(ph, condition)),
                    out_arr[:, 2, :])
            np.save(os.path.join(data_dir, 'r_values', 'r_outer_gt_ph_{}_{}.npy'.format(ph, condition)),
                    out_arr[:, 3, :])


if __name__ == '__main__':
    data_dir = r'.'
    get_r_vals_all(data_dir)