from colicoords.minimizers import DifferentialEvolution
from colicoords import load, save
import numpy as np


def optimize_all():
    for ph in [10000, 1000, 500]:
        print(f'Photons {ph}')
        m_cells = load('cell_obj/cell_ph_{}_filtered.hdf5'.format(ph))

        print('Measured cells loaded')

        print('binary')
        optimize_cells = m_cells.copy()

        res = optimize_cells.optimize_mp()
        obj_vals = [r.objective_value for r in res]

        np.savetxt('minimize_res/m_cells_ph_{}_binary.txt'.format(ph), obj_vals)
        save('cell_obj/m_cells_ph_{}_filtered_binary.hdf5'.format(ph), optimize_cells)

        print('brightfield')
        optimize_cells = m_cells.copy()

        res = optimize_cells.optimize_mp('brightfield')
        obj_vals = [r.objective_value for r in res]

        np.savetxt('minimize_res/m_cells_ph_{}_brightfield.txt'.format(ph), obj_vals)
        save('cell_obj/m_cells_ph_{}_filtered_brightfield.hdf5'.format(ph), optimize_cells)

        print('storm inner')
        optimize_cells = m_cells.copy()

        res = optimize_cells.optimize_mp('storm_inner')
        obj_vals = [r.objective_value for r in res]

        np.savetxt('minimize_res/m_cells_ph_{}_storm.txt'.format(ph), obj_vals)
        save('cell_obj/m_cells_ph_{}_filtered_storm_inner.hdf5'.format(ph), optimize_cells)


if __name__ == '__main__':
    optimize_all()
