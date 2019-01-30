import tifffile
import numpy as np
import matplotlib.pyplot as plt
from colicoords.synthetic_data import add_readout_noise
from scipy.stats import moment
from smitsuite import generate_background

#100 ms expt, EMGain 4
# gain 4, counts to photons, divide by: 0.3682782513636919
empty_pth = r'D:\data\20181107_rhob_cy5\mg1655_100ms_3mW_g4_50\_514_01_\stack1'

empty = tifffile.imread('darkfield.tif')
df = np.mean(empty)
im0 = empty[0]
im0_bg = generate_background(im0, gaussian_kernel=30)
corrected = im0 - im0_bg
zerod = corrected - corrected.mean()
added = zerod + df
plt.imshow(im0_bg)
plt.show()

print(np.mean(added))
print(np.std(added))
print('measured moments', [moment(added, moment=n+1, axis=None) for n in range(5)])


synth = np.ones((512, 512))*df
noise = add_readout_noise(synth, 20)
print(np.mean(noise))
print(np.std(noise))
print([moment(noise, moment=n+1, axis=None) for n in range(5)])

fig, axes = plt.subplots(1, 2)
axes[0].imshow(added)
axes[1].imshow(noise)
plt.show()

pth = r'D:\_processed_data\2018\20181107_rhob_cy5_controls_NHS\mg1655_100ms_3mW_g4_50\mg1655_100ms_3mW_g4_50_1_bf.tif'
img = tifffile.imread(pth)
print(np.median(img)) # 31032


print(np.mean(img))
print(np.std(img))
print('measured moments', [moment(img, moment=n+1, axis=None) for n in range(5)])
# black: 30000
# white: 33000

plt.imshow(img[0])
plt.show()


