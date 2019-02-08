import numpy as np
from colicoords import load, CellFit
import os
from tqdm.auto import tqdm


#https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
def kabsch(P, Q):
    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


def transform_coords(gt_cell, m_cell):
    """uses kabsch to find the translation of storm localizations from m_cell onto gt_cell, then applies this transformation
    to the midline polynomial and fits the result to a polynomial to obtain tranformed coords
    """
    P = np.stack([m_cell.data.data_dict['storm_inner']['x'], m_cell.data.data_dict['storm_inner']['y']]).T
    Q = np.stack([gt_cell.data.data_dict['storm_inner']['x'], gt_cell.data.data_dict['storm_inner']['y']]).T

    assert P.shape == Q.shape

    x = np.linspace(m_cell.coords.xl, m_cell.coords.xr, num=1000, endpoint=True)
    T = np.stack((x, m_cell.coords.p(x))).T

    P_mean = np.mean(P, axis=0)
    Q_mean = np.mean(Q, axis=0)

    Pt = P - P_mean
    Qt = Q - Q_mean

    U = kabsch(Pt, Qt)

    Tt = T - P_mean
    Tr = np.dot(Tt, U)

    Pr = np.dot(Pt, U)
    Prt = Pr + Q_mean
    Trt = Tr + Q_mean

    xl = Trt.T[0].min()
    xr = Trt.T[0].max()
    r = m_cell.coords.r

    a0, a1, a2 = np.polyfit(Trt.T[0], Trt.T[1], 2)[::-1]

    d = {'a0': a0, 'a1': a1, 'a2': a2, 'xl': xl, 'xr': xr, 'r': r}

    return d


def cell_to_dict(cell):
    return {attr: getattr(cell.coords, attr) for attr in ['a0', 'a1', 'a2', 'r', 'xl', 'xr']}


def get_value(gt_cell, m_cell, data_name):
    d_m = transform_coords(gt_cell, m_cell)
    d_g = cell_to_dict(gt_cell)

    if data_name == 'storm_inner':
        d_g['r'] /= (1.5554007217841803 * 1.314602664567288)
        d_m['r'] = d_g['r']

    # copy the cell object because coords values get changed when calling the objective function
    fit = CellFit(gt_cell.copy(), data_name)

    val_m = fit.fit.objective(**d_m)
    val_g = fit.fit.objective(**d_g)

    return val_m, val_g


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

            result = np.array([get_value(gt, m, condition) for m, gt in tqdm(zip(m_sorted, gt_sorted), total=len(m_sorted))])
            np.savetxt(os.path.join(data_dir, 'obj_values', 'obj_vals_ph_{}_{}.txt'.format(ph, condition)), result)
            np.save(os.path.join(data_dir, 'obj_values', 'obj_vals_ph_{}_{}.npy'.format(ph, condition)), result)

            result = np.array([get_value(gt, m, 'storm_inner') for m, gt in tqdm(zip(m_sorted, gt_sorted), total=len(m_sorted))])

            np.savetxt(os.path.join(data_dir, 'obj_values', 'obj_vals_storm_ph_{}_{}.txt'.format(ph, condition)), result)
            np.save(os.path.join(data_dir, 'obj_values', 'obj_vals_storm_ph_{}_{}.npy'.format(ph, condition)), result)

if __name__ == '__main__':
    data_dir = r'D:\Projects\CC_paper\figure_6_v3'
    get_obj_values_all(data_dir)