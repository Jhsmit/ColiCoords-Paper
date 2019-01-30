from colicoords import load, save
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os

data_dir = r'D:\Projects\CC_paper\figures\Figure_6_synthetic_data'

def mk_tempfiles():
    for ph in [10000, 1000, 500]:
        print('photons', ph)
        brightfield = tifffile.imread(os.path.join(data_dir, f'bf_noise_{ph}_photons.tif'))
        tifffile.imsave(os.path.join(data_dir, 'temp_dir', f'bf_{ph}_photons.fit'), brightfield[:20])

        for condition in ['raw', 'binary', 'brightfield', 'brightfield_DE', 'storm_inner']:
            cells = load(os.path.join(data_dir, 'cell_obj', f'm_cells_ph_{ph}_match_{condition}.hdf5'))
            save(os.path.join(data_dir, 'temp_dir', f'm_cells_ph_{ph}_match_{condition}_s100.hdf5'), cells[:100])

        gt_cells = load(os.path.join(data_dir, 'temp_dir', f'gt_cells_ph_{ph}_match_raw.hdf5'))
        save(os.path.join(data_dir, 'temp_dir', f'gt_cells_ph_{ph}_match_raw.hdf5'), gt_cells[:100])

photons = 10000



mk_tempfiles()