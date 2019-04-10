from colicoords.synthetic_data import add_readout_noise, draw_poisson
from colicoords import load
import numpy as np
import mahotas as mh
from tqdm import tqdm
import os
import tifffile


def chunk_list(l, sizes):
    prev = 0
    for s in sizes:
        result = l[prev:prev+s]
        prev += s
        yield result


def generate_images(cell_list, num_images, cell_per_img, cell_per_img_std, shape):
    nums = np.round(np.random.normal(cell_per_img, cell_per_img_std, num_images)).astype(int)
    nums = nums[nums > 0]

    assert sum(nums) < len(cell_list), 'Not enough cells'
    chunked = [chunk for chunk in tqdm(chunk_list(cell_list, nums))]
    dicts = [generate_image(cells, shape) for cells in tqdm(chunked)]
    out_dict = {}
    for i, d in enumerate(dicts):
        for k, v in d.items():
            if 'storm' in k:
                v['frame'] = i + 1
                if k in out_dict:
                    out_dict[k] = np.append(out_dict[k], v)
                else:
                    out_dict[k] = v
            else:
                if k in out_dict:
                    out_dict[k][i] = v
                else:
                    out_dict[k] = np.zeros((num_images, *shape))
                    out_dict[k][i] = v

    return out_dict


def generate_image(cells, shape, max_dist=5):
    thetas = 360 * np.random.rand(len(cells))
    data_list = [cell.data.rotate(theta) for cell, theta in zip(cells, thetas)]
    assert all([data.names == data_list[0].names for data in data_list]), 'All cells must have the same data elements'
    out_dict = {name: np.zeros(shape) for name, dclass in zip(data_list[0].names, data_list[0].dclasses) if dclass != 'storm'}
    for i, data in enumerate(data_list):
        valid_position = False
        while not valid_position:
            pos_x = int(np.round(shape[1] * np.random.rand()))
            pos_y = int(np.round(shape[0] * np.random.rand()))

            min1 = pos_y - int(np.floor(data.shape[0]))
            max1 = min1 + data.shape[0]

            min2 = pos_x - int(np.floor(data.shape[1]))
            max2 = min2 + data.shape[1]

            # Crop the data for when the cell is on the border of the image
            d_min1 = np.max([0 - min1, 0])
            d_max1 = np.min([data.shape[0] + (shape[0] - pos_y), data.shape[0]])

            d_min2 = np.max([0 - min2, 0])
            d_max2 = np.min([data.shape[1] + (shape[1] - pos_x), data.shape[1]])

            data_cropped = data[d_min1:d_max1, d_min2:d_max2]

            # Limit image position to the edges of the image
            min1 = np.max([min1, 0])
            max1 = np.min([max1, shape[0]])
            min2 = np.max([min2, 0])
            max2 = np.min([max2, shape[1]])

            temp_binary = np.zeros(shape)
            temp_binary[min1:max1, min2:max2] = data_cropped.binary_img
            out_binary = (out_dict['binary'] > 0).astype(int)
            distance_map = mh.distance(1 - out_binary, metric='euclidean')

            if np.any(distance_map[temp_binary.astype(bool)] < max_dist):
                continue

            valid_position = True

        for name in data.names:
            data_elem = data_cropped.data_dict[name]
            if data_elem.dclass == 'storm':
                data_elem['x'] += min2
                data_elem['y'] += min1

                xmax, ymax = shape[1], shape[0]
                bools = (data_elem['x'] < 0) + (data_elem['x'] > xmax) + (data_elem['y'] < 0) + (data_elem['y'] > ymax)
                data_out = data_elem[~bools].copy()
                if name in out_dict:
                    out_dict[name] = np.append(out_dict[name], data_out)
                else:
                    out_dict[name] = data_out

                continue
            elif data_elem.dclass == 'binary':
                out_dict[name][min1:max1, min2:max2] += ((i+1)*data_elem)
            else:
                out_dict[name][min1:max1, min2:max2] += data_elem

    return out_dict


def gen_image_from_storm(storm_table, shape, sigma=1.54, sigma_std=0.3):
    xmax = shape[1]
    ymax = shape[0]
    step = 1
    xi = np.arange(step / 2, xmax, step)
    yi = np.arange(step / 2, ymax, step)

    x_coords = np.repeat(xi, len(yi)).reshape(len(xi), len(yi)).T
    y_coords = np.repeat(yi, len(xi)).reshape(len(yi), len(xi))
    x, y = storm_table['x'], storm_table['y']

    img = np.zeros_like(x_coords)
    intensities = storm_table['intensity']
    sigma = sigma * np.ones_like(x) if not sigma_std else np.random.normal(sigma, sigma_std, size=len(x))
    for _sigma, _int, _x, _y in zip(sigma, intensities, x, y):
        img += _int * np.exp(-(((_x - x_coords) / _sigma) ** 2 + ((_y - y_coords) / _sigma) ** 2) / 2)

    return img


def gen_im(data_dir):
    """Generate microscopy images from a list of cell objects by placing them randomly oriented in the image."""
    cell_list = load(os.path.join(data_dir, 'cell_obj', 'cells_final_selected.hdf5'))

    out_dict = generate_images(cell_list, 1000, 10, 3, (512, 512))

    if not os.path.exists(os.path.join(data_dir, 'images')):
        os.mkdir(os.path.join(data_dir, 'images'))

    np.save(os.path.join(data_dir, 'images', 'binary.npy'), out_dict['binary'])
    np.save(os.path.join(data_dir, 'images', 'brightfield.npy'), out_dict['brightfield'])
    np.save(os.path.join(data_dir, 'images', 'foci_inner.npy'), out_dict['foci_inner'])
    np.save(os.path.join(data_dir, 'images', 'foci_outer.npy'), out_dict['foci_outer'])
    np.save(os.path.join(data_dir, 'images', 'storm_inner.npy'), out_dict['storm_inner'])
    np.save(os.path.join(data_dir, 'images', 'storm_outer.npy'), out_dict['storm_outer'])

    tifffile.imsave(os.path.join(data_dir, 'images', 'binary.tif'), out_dict['binary'])
    tifffile.imsave(os.path.join(data_dir, 'images', 'brightfield.tif'), out_dict['brightfield'])
    tifffile.imsave(os.path.join(data_dir, 'images', 'foci_inner.tif'), out_dict['foci_inner'])
    tifffile.imsave(os.path.join(data_dir, 'images', 'foci_outer.tif'), out_dict['foci_outer'])
    np.savetxt(os.path.join(data_dir, 'images', 'storm_inner.txt'), out_dict['storm_inner'])
    np.savetxt(os.path.join(data_dir, 'images', 'storm_outer.txt'), out_dict['storm_inner'])


def noise_bf(data_dir):
    """add poissonian and readout noise to brightfield images"""
    noise = 20
    img_stack = np.load(os.path.join(data_dir, 'images', 'brightfield.npy'))
    for photons in [10000, 1000, 500]:
        ratio = 1.0453  # ratio between 'background' (no cells) and cell wall
        img = (photons*(ratio-1))*img_stack + photons
        img = draw_poisson(img)
        img = add_readout_noise(img, noise)
        tifffile.imsave(os.path.join(data_dir, 'images', 'bf_noise_{}_photons.tif'.format(photons)), img)
        np.save(os.path.join(data_dir, 'images', 'bf_noise_{}_photons.npy'.format(photons)), img)


if __name__ == '__main__':
    np.random.seed(42)
    data_dir = r'.'
    if not os.path.exists(os.path.join(data_dir, 'images')):
        os.mkdir(os.path.join(data_dir, 'images'))
    gen_im(data_dir)
    noise_bf(data_dir)
