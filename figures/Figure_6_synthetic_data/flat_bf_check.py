import tifffile
import numpy as np
import matplotlib.pyplot as plt
from colicoords.synthetic_data import add_readout_noise
from scipy.stats import moment
from smitsuite import generate_background
import seaborn as sns
sns.set_style('white')

#100 ms expt, EMGain 4
# gain 4, counts to photons, divide by: 0.3682782513636919

pth = r'D:\_processed_data\2018\20181107_rhob_cy5_controls_NHS\mg1655_100ms_3mW_g4_50\mg1655_100ms_3mW_g4_50_1_bf_flat.tif'
img = tifffile.imread(pth)
img = img[0][:200, :]
print(np.median(img)) # 31032
print(np.mean(img))
print(np.max(img), np.min(img))
print(np.std(img))
print('measured moments', [moment(img, moment=n+1, axis=None) for n in range(5)])
# black: 30000
# white: 33000

plt.imshow(img, cmap='gray')
plt.show()


