import os
import gzip

import numpy as np
from sklearn.model_selection import train_test_split


class FashionMNIST:

    def __init__(self, path):
        self.path = path
        self.X_train, self.y_train = self.load_fashion()
        self.X_test, self.y_test = self.load_fashion(kind='t10k')
        self.X_val, self.y_val = None, None

    def __repr__(self):
        message = 'Fashion MNIST data.\n' \
                  'Train set Images: {}, Labels: {}\n'.format(self.X_train.shape, self.y_train.shape) \

        if self.X_val is not None and self.y_val is not None:
            message += 'Validation set Images: {}, Labels: {}\n'.format(self.X_val.shape, self.y_val.shape)
        else:
            message += 'Validation set Images: None, Labels: None\n'

        message += 'Test set Images: {}, Labels: {}'.format(self.X_test.shape, self.y_test.shape)

        return message

    def load_fashion(self, kind='train'):
        """
        Load Fashion MNIST data.

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
        labels_path = os.path.join(self.path, '%s-labels-idx1-ubyte.gz' % kind)
        images_path = os.path.join(self.path, '%s-images-idx3-ubyte.gz' % kind)

        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        return images, labels

    def create_validation(self, val_size=0.33, random_state=42):
        """Create validation set."""
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(self.X_train, self.y_train, test_size=val_size, random_state=random_state)

    def get_data(self):
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
