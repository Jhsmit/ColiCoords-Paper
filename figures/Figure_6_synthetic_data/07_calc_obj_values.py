import numpy as np
from colicoords import load, CellFit
import os
from tqdm.auto import tqdm


def cell_to_dict(cell):
    return {attr: getattr(cell.coords, attr) for attr in ['a0', 'a1', 'a2', 'r', 'xl', 'xr']}


def get_value(gt_cell, m_cell):
    d_m = cell_to_dict(m_cell)
    d_g = cell_to_dict(gt_cell)

    d_g['r'] /= (1.5554007217841803 * 1.314602664567288)
    d_m['r'] = d_g['r']

    # copy the cell object because coords values get changed when calling the objective function
    fit_gt = CellFit(gt_cell.copy(), 'storm_inner')
    val_gt = fit_gt.fit.objective(**d_g)

    fit_m = CellFit(m_cell.copy(), 'storm_inner')
    val_m = fit_m.fit.objective(**d_m)

    return val_m, val_gt


def get_value_fit(gt_cell, m_cell):
    d_m = cell_to_dict(m_cell)
    d_g = cell_to_dict(gt_cell)

    d_g['r'] /= (1.5554007217841803 * 1.314602664567288)
    d_m['r'] = d_g['r']

    # copy the cell object because coords values get changed when calling the objective function
    fit_gt = CellFit(gt_cell.copy(), 'storm_inner')
    val_gt = fit_gt.fit.objective(**d_g)

    fit_m = CellFit(m_cell.copy(), 'storm_inner')
    val_m = fit_m.fit.objective(**d_m)

    res = fit_m.fit_parameters('r')
    d_m['r'] = res.params['r']
    val_m_new = fit_m.fit.objective(**d_m)


    return val_m_new, val_gt


def get_obj_values_all(data_dir):
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

            result = np.array([get_value(gt, m) for m, gt in tqdm(zip(m_sorted, gt_sorted), total=len(m_sorted))])

            np.savetxt(os.path.join(data_dir, 'obj_values', 'obj_vals_storm_ph_{}_{}.txt'.format(ph, condition)), result)
            np.save(os.path.join(data_dir, 'obj_values', 'obj_vals_storm_ph_{}_{}.npy'.format(ph, condition)), result)

            # result = np.array([get_value_fit(gt, m) for m, gt in tqdm(zip(m_sorted, gt_sorted), total=len(m_sorted))])
            #
            # np.savetxt(os.path.join(data_dir, 'obj_values_new', 'obj_vals_fit_storm_ph_{}_{}.txt'.format(ph, condition)), result)
            # np.save(os.path.join(data_dir, 'obj_values_new', 'obj_vals_fit_storm_ph_{}_{}.npy'.format(ph, condition)), result)


if __name__ == '__main__':
    data_dir = r'D:\Projects\CC_paper\figure_6_v3'
    get_obj_values_all(data_dir)