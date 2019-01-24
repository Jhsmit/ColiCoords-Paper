from colicoords import load, save
from tqdm.auto import tqdm
import numpy as np
import fastcluster as fc
from scipy.cluster.hierarchy import fcluster


def filter_cells(m_names, gt_names, m_cells, gt_cells, max_d=3):
    """Corrects cells with too many STORM localizations and removes cells with too few"""

    m_remove = []
    gt_remove = []
    for m_name, gt_name in tqdm(zip(m_names, gt_names), total=len(m_names)):
        m_i = m_cells.name.tolist().index(m_name)
        gt_i = gt_cells.name.tolist().index(gt_name)

        m_c = m_cells[m_i]
        gt_c = gt_cells[gt_i]

        for elem_name in ['storm_inner', 'storm_outer']:
            if len(m_c.data.data_dict[elem_name]) > len(gt_c.data.data_dict[elem_name]):
                st_elem = m_c.data.data_dict[elem_name]
                X = np.array([st_elem['x'], st_elem['y']]).T.copy()
                linkage = fc.linkage(X)
                clusters = fcluster(linkage, max_d, criterion='distance')
                counts = np.bincount(clusters)
                i_max = np.argmax(counts)

                b = [clusters == i_max]

                m_c.data.data_dict[elem_name] = m_c.data.data_dict[elem_name][b].copy()

                try:
                    assert len(m_c.data.data_dict[elem_name]) == len(gt_c.data.data_dict[elem_name])
                except AssertionError:
                    m_remove.append(m_name)
                    gt_remove.append(gt_name)

            elif len(m_c.data.data_dict[elem_name]) < len(gt_c.data.data_dict[elem_name]):
                m_remove.append(m_name)
                gt_remove.append(gt_name)

    m_final = list([name + '\n' for name in m_names if name not in m_remove])
    gt_final = list([name + '\n' for name in gt_names if name not in gt_remove])

    return m_final, gt_final, m_cells, gt_cells


def filter_all():
    gt_cells = load('cell_obj/cells_final_selected.hdf5')

    for ph in [10000, 1000, 500]:
        print('Photons', ph)
        with open(f'matched_names/m_cells_ph_{ph}_match.txt', 'r') as f:
            m_names = f.readlines()

        m_names = list([n.rstrip() for n in m_names])
        m_cells = load(f'cell_obj/cell_ph_{ph}_raw.hdf5')

        with open(f'matched_names/gt_cells_ph_{ph}_match.txt', 'r') as f:
            gt_names = f.readlines()

        gt_names = list([n.rstrip() for n in gt_names])

        m_final, gt_final, m_cells, gt_cells = filter_cells(m_names, gt_names, m_cells, gt_cells)

        with open('matched_names/gt_cells_ph_{}_match_filter.txt'.format(ph), 'w') as f:
            f.writelines(gt_final)

        with open('matched_names/m_cells_ph_{}_match_filter.txt'.format(ph), 'w') as f:
            f.writelines(m_final)

        for i, (m_, gt_) in tqdm(enumerate(zip(m_final, gt_final))):
            m_i = m_cells.name.tolist().index(m_.rstrip())
            g_i = gt_cells.name.tolist().index(gt_.rstrip())

            try:
                assert len(m_cells[m_i].data.data_dict['storm_inner']) == len(
                    gt_cells[g_i].data.data_dict['storm_inner'])
            except AssertionError:
                print('Assertion error:', i)

        save(f'cell_obj/cell_ph_{ph}_filtered.hdf5', m_cells)


if __name__ == '__main__':
    filter_all()
