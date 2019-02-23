import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'axes.titlesize': 1})

import numpy as np
import os
import random

def plot_compare2(syn_batch, ref_batch, batch_path, i, vmin=0, vmax=255, scale=True):
    number_of_plots = np.size(syn_batch, 0)
    if not os.path.exists(batch_path):
        os.makedirs(batch_path)
    else:
        for the_file in os.listdir(batch_path):
            file_path = os.path.join(batch_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    syn_images = syn_batch
    real_images = ref_batch
    diff = syn_images - real_images
    if syn_images.shape[-1] == 1:
        syn_images = np.reshape(syn_images, newshape=syn_images.shape[:-1])
    if real_images.shape[-1] == 1:
        real_images = np.reshape(real_images, newshape=real_images.shape[:-1])
    if diff.shape[-1] == 1:
        diff = np.reshape(diff, newshape=diff.shape[:-1])

    for plot_number in range(number_of_plots):
        figure_name = 'epoch_{}_'.format(i) + 'image_{}.png'.format(plot_number)
        _, ax = plt.subplots(1, 3, sharex=True, sharey=True, squeeze=False)
        x = syn_images[plot_number]
        y = real_images[plot_number]
        z = diff[plot_number]
        if scale:
            x = x + max(-np.min(x), 0)
            y = y + max(-np.min(y), 0)
            z = z + max(-np.min(z), 0)

            x_max = np.max(x)
            y_max = np.max(y)
            z_max = np.max(z)

            if x_max != 0:
                x /= x_max
            x *= 255

            if y_max != 0:
                y /= y_max
            y *= 255

            if z_max != 0:
                z /= z_max
            z *= 255
        ax[0][0].imshow(x.astype('uint8'), vmin=vmin, vmax=vmax, interpolation='lanczos', cmap='gray')
        ax[0][0].set_title('Syntetic',fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax[0][0].set_axis_off()

        ax[0][1].imshow(y.astype('uint8'), vmin=vmin, vmax=vmax, interpolation='lanczos', cmap='gray')
        ax[0][1].set_title('Refined', fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax[0][1].set_axis_off()

        ax[0][2].imshow(z.astype('uint8'), vmin=vmin, vmax=vmax, interpolation='lanczos', cmap='gray')
        ax[0][2].set_title('diff', fontdict={'fontsize': 8, 'fontweight': 'medium'})
        ax[0][2].set_axis_off()
        plt.savefig(os.path.join(batch_path, figure_name), dpi=600)
        plt.close()