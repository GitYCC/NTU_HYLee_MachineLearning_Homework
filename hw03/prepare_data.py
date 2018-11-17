import os
import tarfile
import pickle
from six.moves.urllib.request import urlretrieve

import numpy as np
from progressbar import ProgressBar


DIR_RAW_CIFAR10 = 'cifar-10-batches-py'
NUM_LABLE_EACH_CLASS = 500
NUM_UNLABLE_EACH_CLASS = 4500
NUM_CLASS = 10
progressbar = [None]


def _show_progress(count, block_size, total_size):
    if progressbar[0] is None:
        progressbar[0] = ProgressBar(maxval=total_size)

    downloaded = block_size * count
    if downloaded <= total_size:
        progressbar[0].update(downloaded)
    else:
        progressbar[0].finish()
        progressbar[0] = None


def _unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def maybe_download_cifar10(folder):
    path = os.path.join(folder, 'cifar-10-python.tar.gz')
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    if not os.path.exists(path):
        print('download from {}'.format(url))
        filename, _ = urlretrieve(url, path, _show_progress)


def tar_cifar10(folder):
    with tarfile.open(os.path.join(folder, 'cifar-10-python.tar.gz'), 'r') as tar:
        tar.extractall(folder)


def create_label_and_unlabel_pickle(folder):
    images = None
    labels = list()
    batch_files = sorted(
        filter(
            lambda x: 'data_batch_' in x,
            os.listdir(os.path.join(folder, DIR_RAW_CIFAR10))
        )
    )
    for batch_file in batch_files:
        path = os.path.join(folder, DIR_RAW_CIFAR10, batch_file)
        content = _unpickle(path)
        if images is None:
            images = content['data']
        else:
            images = np.concatenate((images, content['data']), axis=0)
        labels += content['labels']

    counters = {i: 0 for i in range(NUM_CLASS)}
    label_masks = {i: list() for i in range(NUM_CLASS)}
    unlabel_masks = list()

    for i in range((NUM_LABLE_EACH_CLASS + NUM_UNLABLE_EACH_CLASS) * NUM_CLASS):
        label = labels[i]
        if counters[label] < NUM_LABLE_EACH_CLASS:
            label_masks[label].append(i)
        else:
            unlabel_masks.append(i)
        counters[label] += 1

    with open(os.path.join(folder, 'all_label.p'), 'wb') as fw:
        label_data = np.concatenate([images[label_masks[i]] for i in range(NUM_CLASS)], axis=0)
        pickle.dump(label_data, fw, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(folder, 'all_unlabel.p'), 'wb') as fw:
        unlabel_data = images[unlabel_masks]
        pickle.dump(unlabel_data, fw, pickle.HIGHEST_PROTOCOL)


def create_test_pickle(folder):
    path = os.path.join(folder, DIR_RAW_CIFAR10, 'test_batch')
    content = _unpickle(path)

    with open(os.path.join(folder, 'test.p'), 'wb') as fw:
        test_data = content['data']
        pickle.dump(test_data, fw, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(folder, 'test_ans.txt'), 'w') as fw:
        test_labels = content['labels']
        for label in test_labels:
            fw.write(str(label) + '\n')


def main():
    folder = './data'
    maybe_download_cifar10(folder)
    tar_cifar10(folder)
    create_label_and_unlabel_pickle(folder)
    create_test_pickle(folder)


if __name__ == '__main__':
    main()
