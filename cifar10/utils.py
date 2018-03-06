import numpy as np
from matplotlib import pyplot as plt
import csv
import math
from PIL import Image


def plot_log(filename, show=True):
    """ https://github.com/XifengGuo/CapsNet-Keras/blob/master/utils.py
    """

    # load data
    keys = []
    values = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:,0] += 1

    fig = plt.figure(figsize=(4,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find('acc') >= 0:  # acc
            plt.plot(values[:, 0], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    # fig.savefig('result/log.png')
    if show:
        plt.show()


def stack_images_two_arrays(x_augmented, x_recon, rows, cols):
    """ Stack images together and return the image for two arrays.
        So the first row shows the first array and the second the second etc.
    """
    width = x_augmented[0].shape[0]
    height = x_augmented[0].shape[1]
    stacked_img = Image.new('RGB', (rows*(width+5), cols*(height+5)))
    for i in range(rows):
        for j in range(cols):
            img_augmented = Image.fromarray((x_augmented[i*rows + j]*255).astype(np.uint8))
            img_recon = Image.fromarray((x_recon[i*rows + j]*255).astype(np.uint8))
            
            pos = (j * (width+5), i * (height+5) * 2)
            stacked_img.paste(img_augmented, pos)
            stacked_img.paste(img_recon, (pos[0], pos[1]+height))

    return stacked_img


def stack_images(x, cols):
    """ Stack images by row together and return the image.
    """
    width = x[0].shape[0]
    height = x[0].shape[1]
    rows = int(len(x) / cols)

    stacked_img = Image.new('RGB', (cols*(height), rows*(width)))
    for i in range(rows):
        for j in range(cols):
            data = x[i * cols + j]
            img = Image.fromarray((data * 255).astype(np.uint8))
            pos = (j * (width), i * (height))
            stacked_img.paste(img, pos)

    return stacked_img


def center_crop(x, center_crop_size, **kwargs):
    """ From https://github.com/keras-team/keras/issues/3338
    """
    centerw, centerh = x.shape[1]//2, x.shape[2]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[:, centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh, :]

# width, height = 24, 24 
# width_crop_left = 8
# height_crop_top = 8
# x_batch = x_batch[:, width_crop_left:width+width_crop_left, height_crop_top:height+height_crop_top, :]

def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
    """ From https://github.com/keras-team/keras/issues/3338
    """
    np.random.seed(sync_seed)
    w, h = x.shape[1], x.shape[2]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[:, offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1], :]