from colicoords.cnn.unet import get_unet_256
from colicoords.cnn.preprocess import DefaultImgSequence, resize_stack
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import keras
import tensorflow
import os


data_dir = '.'
if not os.path.exists(os.path.join(data_dir, 'wts')):
    os.mkdir(os.path.join(data_dir, 'wts'))

if not os.path.exists(os.path.join(data_dir, 'logs')):
    os.mkdir(os.path.join(data_dir, 'logs'))

binary = np.load(os.path.join(data_dir, 'images', 'binary.npy'))
binary_resized = resize_stack((binary > 0).astype(int), 0.5, img_type='binary')
del binary

with open('CNN_versions.txt', 'w') as f:
    f.write('Keras: ' + keras.__version__ + '\n')
    f.write('Tensorflow: ' + tensorflow.__version__ + '\n')


for ph in [10000, 1000, 500]:
    print('{} photons'.format(ph))
    bf = np.load(os.path.join(data_dir, 'images', 'bf_noise_{}_photons.npy'.format(ph)))
    brightfield_resized = resize_stack(bf, 0.5)[:400]

    isq = DefaultImgSequence(brightfield_resized, binary_resized[:400])
    vsq, tsq = isq.val_split(1 / 8., random=True)
    print(len(tsq), len(vsq))

    model = get_unet_256(input_shape=(256, 256, 1))
    cp = ModelCheckpoint(os.path.join(data_dir, 'wts', 'wts_bf_' + str(ph) + '_photons_{epoch:02d}-{val_loss:.4f}.h5'), monitor='val_loss',
                         save_weights_only=True, verbose=1, mode='min')
    tb = TensorBoard(log_dir=os.path.join(data_dir, 'logs', 'run_photons_{}'.format(ph)))
    model.fit_generator(tsq, steps_per_epoch=len(tsq), epochs=50, validation_data=vsq, callbacks=[cp, tb])