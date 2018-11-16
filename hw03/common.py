#!python2.7
import sys
import os
import pickle
import random

from PIL import Image
import numpy as np
import keras
from matplotlib import pyplot as plt
from IPython.display import clear_output


def load_label(folder):
    with open(os.path.join(folder, 'all_label.p'), 'rb') as fr:
        label_data = pickle.load(fr)
    label_data = np.array(label_data, dtype='uint8')
    # preproc
    nb_class = 10
    nb_pic_in_class = 500
    height = 32
    width = 32
    nb_channel = 3
    X = label_data
    X.shape = (nb_class*nb_pic_in_class, nb_channel, height, width)
    from keras.utils import to_categorical
    Y = to_categorical([i for i in range(nb_class) for j in range(nb_pic_in_class)])
    return (X, Y)


def load_unlabel(folder):
    with open(os.path.join(folder, 'all_unlabel.p'), 'rb') as fr:
        unlabel_data = pickle.load(fr)

    unlabel_data = np.array(unlabel_data)
    # preproc
    nb_pic = unlabel_data.shape[0]
    height = 32
    width = 32
    nb_channel = 3
    X = unlabel_data
    X.shape = (nb_pic, nb_channel, height, width)
    return X


def load_test(folder):
    with open(os.path.join(folder, 'test.p'), 'rb') as fr:
        test_data = pickle.load(fr)

    test_data = np.array(test_data)
    return test_data


def data_augmentation(X, Y):  # noqa: N803
    # mirror
    mirrX = np.flip(X, axis=3)
    newX = np.concatenate((X, mirrX), axis=0)
    newY = np.concatenate((Y, Y), axis=0)

    return (newX, newY)


def split_data(X, Y, ratio=0.9):  # noqa: N803
    n = X.shape[0]
    if n != Y.shape[0]:
        raise ValueError('number of data is not same between X and Y')
    n_train = int(n*ratio)

    ind_random = [i for i in range(n)]
    random.shuffle(ind_random)

    ind_train = ind_random[:n_train]
    ind_valid = ind_random[n_train:]

    X_train = np.take(X, ind_train, axis=0)
    Y_train = np.take(Y, ind_train, axis=0)
    X_valid = np.take(X, ind_valid, axis=0)
    Y_valid = np.take(Y, ind_valid, axis=0)
    return (X_train, Y_train, X_valid, Y_valid)


def draw(ndarray):
    ndarray = np.expand_dims(ndarray, axis=0)
    rgb = transform_channel(ndarray, 'channels_first')
    rgb = rgb[0]
    return Image.fromarray(rgb, 'RGB')


def transform_channel(ndarray, orig_mode):
    # orig_mode: 'channels_first' or 'channels_last'
    if orig_mode == 'channels_first':
        r = np.expand_dims(ndarray[:, 0, :, :], axis=-1)
        g = np.expand_dims(ndarray[:, 1, :, :], axis=-1)
        b = np.expand_dims(ndarray[:, 2, :, :], axis=-1)
        rgb = np.concatenate((r, g, b), axis=-1)
        return rgb
    elif orig_mode == 'channels_last':
        r = np.expand_dims(ndarray[:, :, :, 0], axis=1)
        g = np.expand_dims(ndarray[:, :, :, 1], axis=1)
        b = np.expand_dims(ndarray[:, :, :, 2], axis=1)
        rgb = np.concatenate((r, g, b), axis=1)
        return rgb


class PlotLosses(keras.callbacks.Callback):
    def __init__(self, output_img=None):
        super(keras.callbacks.Callback, self).__init__()
        self.output_img = output_img

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()
        self.__plot()
        plt.legend()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        self.__plot()

    def on_train_end(self, logs={}):
        plt.close()

    def __plot(self):
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label='loss', color='red', marker='.')
        plt.plot(self.x, self.val_losses, label='val_loss', color='green', marker='.')
        plt.draw()
        if self.output_img:
            plt.savefig(self.output_img)
        plt.pause(0.001)


class Tee():
    def __init__(self, name, mode):
        self.__file = open(name, mode)
        self.__stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        """Remove Tee."""
        sys.stdout = self.__stdout
        self.__file.flush()
        self.__file.close()

    def write(self, data):
        self.__file.write(data)
        self.__stdout.write(data)

    def flush(self):
        self.__file.flush()
        self.__stdout.flush()
