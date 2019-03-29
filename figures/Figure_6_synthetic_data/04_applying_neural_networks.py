from colicoords.cnn.unet import get_unet_256
from colicoords.cnn.preprocess import resize_stack, norm_hampel
import numpy as np
import tifffile
import os

data_dir = '.'

wts = [
    'wts_bf_10000_photons_50-0.0076.h5',
    'wts_bf_1000_photons_50-0.0261.h5',
    'wts_bf_500_photons_49-0.0482.h5'
]


for ph, wt in zip([10000, 1000, 500], wts):
    bf = np.load(os.path.join(data_dir, 'images', 'bf_noise_{}_photons.npy'.format(ph)))
    brightfield_resized = resize_stack(bf, 0.5)
    del bf

    bf_norm = np.stack([norm_hampel(arr) for arr in brightfield_resized])

    model = get_unet_256(input_shape=(256, 256, 1))
    model.load_weights(os.path.join(data_dir, 'wts', wt))
    prediction = model.predict(np.expand_dims(bf_norm, -1))

    predict_resized = resize_stack(prediction.squeeze(), 2)

    tifffile.imsave(os.path.join(data_dir, 'images', 'binary_{}photons_predicted.tif'.format(ph)),
                    (predict_resized > 0.5).astype(int))
