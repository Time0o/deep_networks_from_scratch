import html
import json
import os
import textwrap
from functools import partial
from glob import glob

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import scipy.io


BATCH_META_FMT = 'batches.meta.mat'
DATA_BATCH_FMT = 'data_batch_*.mat'
TEST_BATCH_FMT = 'test_batch.mat'


def _to_image(vector):
    return vector.reshape(3, 32, 32).transpose(1, 2, 0)


def _to_vector(img):
    return img.transpose(2, 0, 1).reshape(-1)


def _to_display(vector, normalize=False):
    img = _to_image(vector)

    if normalize:
        img = img / 255

    return img


def _remove_ticks(ax):
    ax.tick_params(axis='both',
                   which='both',
                   bottom=False,
                   top=False,
                   left=False,
                   right=False,
                   labelbottom=False,
                   labelleft=False)


class Dataset:
    def __init__(self, X, Y, y, labels=None, add_bias=False, visual=True):
        if add_bias:
            X = np.vstack((X, np.ones((1, X.shape[1]))))

        self.X = X
        self.Y = Y
        self.y = y
        self.labels = labels

        self.n = X.shape[1]
        self.input_size = X.shape[0]
        self.num_classes = Y.shape[0]

        self.visual = visual

    def shuffle(self):
        i = np.random.permutation(self.n)

        return Dataset(self.X[:, i], self.Y[:, i], self.y[:, i])

    def batch(self, start, end):
        return Dataset(self.X[:, start:end],
                       self.Y[:, start:end],
                       self.y[:, start:end])

    def subsample(self, dims=None, n=None):
        dims = dims if dims is not None else self.input_size
        n = n if n is not None else self.n

        return Dataset(self.X[:dims, :n], self.Y[:dims, :n], self.y[:dims, :n])

    def bag(self):
        i = np.random.choice(self.n, size=self.n)

        return Dataset(self.X[:, i], self.Y[:, i], self.y[:, i])

    def join(self, ds):
        X = np.concatenate((self.X, ds.X), axis=1)
        Y = np.concatenate((self.Y, ds.Y), axis=1)
        y = np.concatenate((self.y, ds.y), axis=1)

        return Dataset(X, Y, y)

    def augment_color(self, jitter_ratio=1, verbose=False):
        if not self.visual:
            raise ValueError("can only augment image data")

        # to avoid trouble, only implement this for the specific case of
        # Cifar images without added bias row
        assert self.input_size == 3072

        aug = mx.image.ColorJitterAug(
            brightness=jitter_ratio,
            contrast=jitter_ratio,
            saturation=jitter_ratio)

        X_aug = np.empty_like(self.X)

        for k in range(self.n):
            if verbose:
                fmt = "{}/{}".format(k + 1, self.n)

                end = '\n' if k == self.n - 1 else ''
                print(fmt.ljust(80) + "\r", end=end, flush=True)

            img = _to_display(self.X[:, k])

            img = aug(mx.nd.array(img)).asnumpy()

            img = (img - img.min()) / (img.max() - img.min())

            X_aug[:, k] = img.transpose(2, 0, 1).reshape(self.input_size)

        return Dataset(X_aug, self.Y, self.y)

    def augment_orientation(self, flip=True, rot=False, verbose=False):
        X_aug = self.X.copy()

        if rot:
            X_rot = X_aug.copy()
            for _ in range(X_rot.shape[1]):
                if verbose:
                    print("rotating...")

                for k in range(self.n):
                    X_rot[:, k] = _to_vector(np.rot90(_to_image(X_rot[:, k])))

                X_aug = np.hstack((X_aug, X_rot))

        if flip:
            if verbose:
                print("flipping")

            X_flip = X_aug.copy()
            for k in range(X_flip.shape[1]):
                X_flip[:, k] = _to_vector(np.fliplr(_to_image(X_flip[:, k])))

            X_aug = np.hstack((X_aug, X_flip))

        new_n = X_aug.shape[1]

        new_Y = np.tile(self.Y, new_n // self.n)
        new_y = np.tile(self.y, new_n // self.n)

        return Dataset(X_aug, new_Y, new_y)

    def preview(self, w=3, h=3, shuffle=False):
        if not self.visual:
            raise ValueError("can only preview image data")

        _, axes = plt.subplots(h, w, figsize=(8, 8 * (h / w)))

        X = self.X

        if shuffle:
            i = np.random.permutation(X.shape[1])
            X = X[:, i]

        # display images
        for i in range(h):
            for j in range(w):
                axes[i, j].imshow(_to_display(X[:, i * w + j]))

                _remove_ticks(axes[i, j])

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)


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
                             add_bias=False,
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

        ds = partial(Dataset, labels=self._labels, add_bias=add_bias)

        return ds(data_train, labels_cat_train, labels_train), \
               ds(data_val, labels_cat_val, labels_val), \
               ds(data_test, labels_cat_test, labels_test)

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
                axes[i, j].imshow(_to_display(data[:, k], normalize=True))

                _remove_ticks(axes[i, j])

            axes[i, 0].set_ylabel(label, labelpad=60, rotation=0, size='large')

        plt.tight_layout(pad=0, w_pad=0, h_pad=0)


class Text:
    def __init__(self, text, characters=None):
        self.text = text

        if characters is None:
            self.characters = sorted(set(self.text))
        else:
            self.characters = characters

        self.num_characters = len(self.characters)

        self._char_to_ind = {c: i for i, c in enumerate(self.characters)}
        self._ind_to_char = {i: c for i, c in enumerate(self.characters)}

    @classmethod
    def from_file(cls, data_dir, filename):
        with open(os.path.join(data_dir, filename), 'r') as f:
            text = f.read()

        return cls(text)

    def get_index(self, char, one_hot=False):
        ind = self._char_to_ind[char]

        if one_hot:
            return np.eye(self.num_characters, dtype=int)[ind][:, np.newaxis]
        else:
            return ind

    def get_character(self, ind, one_hot=False):
        if one_hot:
            ind = np.argmax(ind)

        return self._ind_to_char[ind]

    def get_indices(self, chars, one_hot=False):
        inds = [self._char_to_ind[c] for c in chars]

        if one_hot:
            return np.eye(self.num_characters, dtype=int)[inds].T
        else:
            return inds

    def get_characters(self, inds, one_hot=False):
        if one_hot:
            inds = np.argmax(inds, axis=0)

        return ''.join([self._ind_to_char[i] for i in inds])

    def sequence(self, beg, end, rep='characters', labeled=False):
        if labeled:
            assert rep == 'indices_one_hot'

            X = self._sequence(beg, min(end, len(self.text) - 1), rep=rep)
            Y = self._sequence(beg + 1, min(end + 1, len(self.text)), rep=rep)

            return Dataset(X, Y, Y.argmax(axis=0), visual=False)
        else:
            return self._sequence(beg, end, rep=rep)

    def _sequence(self, beg, end, rep):
        seq = self.text[beg:end]

        if rep == 'characters':
            return seq
        elif rep == 'indices':
            return self.get_indices(seq)
        elif rep == 'indices_one_hot':
            return self.get_indices(seq, one_hot=True)
        else:
            raise ValueError("invalid 'rep'")

        return seq


class TrumpTweetArchive:
    START_CHARACTER = '\t'
    STOP_CHARACTER = '\n'

    def __init__(self, data_dir, file_fmt):
        # get list of tweet files
        batches = sorted(glob(os.path.join(data_dir, file_fmt)))

        # read in tweets
        tweets = []
        for batch in batches:
            with open(batch, 'r') as f:
                tweets += [
                    self._preprocess_tweet(t['text'])
                    for t in json.load(f)
                ]

        self.characters = set(''.join(tweets)) | \
                          {self.START_CHARACTER, self.STOP_CHARACTER}

        self.num_characters = len(self.characters)

        self.tweets = [
            Text(tweet, characters=self.characters)
            for tweet in tweets
        ]

        self.num_tweets = len(self.tweets)

    def get_index(self, char, one_hot=False):
        return self.tweets[0].get_index(char, one_hot=one_hot)

    def get_character(self, ind, one_hot=False):
        return self.tweets[0].get_character(ind, one_hot=one_hot)

    def get_indices(self, chars, one_hot=False):
        return self.tweets[0].get_indices(chars, one_hot=one_hot)

    def get_characters(self, inds, one_hot=False):
        return self.tweets[0].get_characters(inds, one_hot=one_hot)

    def get_start_character(self, rep='character'):
        return self._get_special_character(self.START_CHARACTER, rep=rep)

    def get_stop_character(self, rep='character'):
        return self._get_special_character(self.STOP_CHARACTER, rep=rep)

    def random_preview(self, n=10, wrap=50):
        i = np.random.choice(range(len(self.tweets)), size=n, replace=False)

        for j in i:
            text = self.tweets[j].text
            text = text.encode('unicode_escape').decode('utf-8')

            print('\n'.join(textwrap.wrap(text)) + '\n')

    def _preprocess_tweet(self, tweet):
        # replace tabs and newlines by spaces
        tweet = tweet.replace('\t', ' ').replace('\n', ' ')

        # filter out non-ascii characters
        tweet = ''.join(filter(lambda c: ord(c) >= 32 and ord(c) <= 126, tweet))

        # decode html sequences
        tweet = html.unescape(tweet)

        # add start and stop characters
        tweet = self.START_CHARACTER + tweet + self.STOP_CHARACTER

        return tweet

    def _get_special_character(self, char, rep):
        if rep == 'character':
            return char
        elif rep == 'index':
            return self.get_index(char)
        elif rep == 'index_one_hot':
            return self.get_index(char, one_hot=True)
        else:
            raise ValueError("invalid 'rep'")
