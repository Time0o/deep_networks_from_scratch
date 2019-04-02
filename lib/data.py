import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


BATCH_META_FMT = 'batches.meta.mat'
DATA_BATCH_FMT = 'data_batch_*.mat'
TEST_BATCH_FMT = 'test_batch.mat'


class Dataset:
    def __init__(self, X, Y, y):
        self.X = X
        self.Y = Y
        self.y = y

        self.n = X.shape[1]
        self.input_size = X.shape[0]
        self.num_classes = Y.shape[0]

    def subsample(self, dims=None, n=None):
        dims = dims if dims is not None else self.input_size
        n = n if n is not None else self.n

        return Dataset(self.X[:dims, :n], self.Y[:dims, :n], self.y[:dims, :n])


class Cifar:
    def __init__(self,
                 data_dir,
                 batch_meta_fmt=BATCH_META_FMT,
                 data_batch_fmt=DATA_BATCH_FMT,
                 test_batch_fmt=TEST_BATCH_FMT,
                 dtype=np.float64):

        self._batch_meta_path = os.path.join(data_dir, batch_meta_fmt)
        self._data_batch_path = os.path.join(data_dir, data_batch_fmt)
        self._test_batch_path = os.path.join(data_dir, test_batch_fmt)
        self._dtype = dtype

        self._load()

    @staticmethod
    def _load_labels(filename):
        mat = scipy.io.loadmat(filename)

        return [l[0] for l in np.squeeze(mat['label_names'])]

    @staticmethod
    def _load_batch(filename, dtype):
        mat = scipy.io.loadmat(filename)

        # extract data
        data = mat['data'].astype(dtype).T

        # extract labels
        labels = mat['labels'].astype(int).T

        # convert labels to categorical representation
        num_labels = len(np.unique(labels))
        labels_cat = np.eye(num_labels, dtype=int)[np.squeeze(labels)].T

        return data, labels_cat, labels

    def _load(self):
        # load labels
        self._labels = self._load_labels(self._batch_meta_path)

        # load training batches
        train_batches = [
            self._load_batch(f, self._dtype)
            for f in sorted(glob(self._data_batch_path))
        ]

        train_data, train_labels_cat, train_labels = zip(*train_batches)

        self._train_data = np.concatenate(train_data, axis=1)
        self._train_labels_cat = np.concatenate(train_labels_cat, axis=1)
        self._train_labels = np.concatenate(train_labels, axis=1)

        # load test batch
        test_batch = self._load_batch(self._test_batch_path, self._dtype)
        self._test_data, self._test_labels_cat, self._test_labels = test_batch

    def labels(self):
        return self._labels

    def train_val_test_split(self,
                             n_val,
                             n_train=None,
                             n_test=None,
                             shuffle=False,
                             normalize='scale'):

        # assert requested data dimensions
        if n_train is None:
            n_train = self._train_data.shape[1] - n_val

        if n_train <= 0 or n_train + n_val > self._train_data.shape[1]:
            raise ValueError("requested more training data then available")

        if n_test is None:
            n_test = self._test_data.shape[1]
        elif n_test > self._test_data.shape[1]:
            raise ValueError("requested more test data then available")

        # aliases
        data = self._train_data
        labels_cat = self._train_labels_cat
        labels = self._train_labels

        # optionally shuffle data
        if shuffle:
            i = np.random.permutation(self._train_data.shape[1])

            data = data[:, i]
            labels_cat = labels_cat[:, i]
            labels = labels[:, i]

        # split data
        def split_batch(batch):
            return batch[:, :n_train], batch[:, n_train:(n_train + n_val)]

        data_train, data_val = split_batch(data)
        labels_cat_train, labels_cat_val = split_batch(labels_cat)
        labels_train, labels_val = split_batch(labels)

        data_test = self._test_data[:, :n_test]
        labels_cat_test = self._test_labels_cat[:, :n_test]
        labels_test = self._test_labels[:, :n_test]

        # normalize data
        if normalize == 'scale':
            data_train = data_train / 255
            data_val = data_val / 255
            data_test = data_test / 255
        elif normalize == 'zscore':
            mean = data_train.mean(axis=1, keepdims=True)
            std = data_train.std(axis=1, keepdims=True)

            data_train = (data_train - mean) / std
            data_val = (data_val - mean) / std
            data_test = (data_test - mean) / std
        else:
            raise ValueError("'normalize' must be either 'scale' or 'zscore'")

        return Dataset(data_train, labels_cat_train, labels_train), \
               Dataset(data_val, labels_cat_val, labels_val), \
               Dataset(data_test, labels_cat_test, labels_test)

    def preview(self, which='train', n=5, shuffle=False):
        _, axes = plt.subplots(len(self._labels), n,
                               figsize=(8, 8 / n * len(self._labels)))

        # choose images from either training or test set
        if which == 'train':
            data, labels = self._train_data, self._train_labels
        elif which == 'test':
            data, labels = self._test_data, self._test_labels
        else:
            raise ValueError("'which' must be either 'train' or 'test'")

        # optionally shuffle images before choosing the ones to display
        if shuffle:
            i = np.random.permutation(data.shape[1])

            data = data[:, i]
            labels = labels[:, i]

        # display images
        for i, label in enumerate(self._labels):
            for j, k in enumerate(np.where(np.squeeze(labels) == i)[0][:n]):
                img = data[:, k].reshape(3, 32, 32).transpose(1, 2, 0) / 255

                axes[i, j].imshow(img)

                axes[i, j].tick_params(axis='both',
                                       which='both',
                                       bottom=False,
                                       top=False,
                                       left=False,
                                       right=False,
                                       labelbottom=False,
                                       labelleft=False)

            axes[i, 0].set_ylabel(label, labelpad=60, rotation=0, size='large')

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
