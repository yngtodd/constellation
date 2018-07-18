import os
import gzip
import numpy as np


def load_mnist(path, kind='train'):
    """
    Load MNIST data.

    Parameters:
    ----------
    * `path` [str]
      Path to the data.

    * `kind` [str]
      Determines whether to load train or test set.
      Options - 'train' or 't10k'

    Returns:
    -------
    * `images` [ndarray]
      Image data
    * `labels` [ndarray]
      Label information

    References:
    ----------
    https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels
