from colicoords import load, save
import numpy as np
import os


data_dir = r'D:\Projects\CC_paper\figures\Figure_6_synthetic_data'

gt = load(os.path.join(data_dir, 'cell_obj', 'gt_cells_ph_10000_match_raw.hdf5'))
save(os.path.join(data_dir, 'temp_dir', 'gt_cells_ph_10000_match_raw_s100.hdf5'), gt[:100])
