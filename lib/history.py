import os
import pickle

import matplotlib.pyplot as plt


class TrainHistory:
    def __init__(self, store_learning_rate=False):
        self.train_loss = []
        self.train_cost = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_cost = []
        self.val_accuracy = []

        if store_learning_rate:
            self.learning_rate = []

        self.final_network = None
        self.title = None

        self.length = 0

    def extend(self, network, ds_train, ds_val):
        train_loss, train_cost = network.cost(ds_train, return_loss=True)
        self.train_loss.append(train_loss)
        self.train_cost.append(train_cost)
        self.train_accuracy.append(network.accuracy(ds_train))

        val_loss, val_cost = network.cost(ds_val, return_loss=True)
        self.val_loss.append(val_loss)
        self.val_cost.append(val_cost)
        self.val_accuracy.append(network.accuracy(ds_val))

        self.length += 1

    def add_learning_rate(self, eta):
        self.learning_rate.append(eta)

    def add_final_network(self, network):
        self.final_network = network

    def add_title(self, title):
        self.title = title

    def visualize(self, axes=None):
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(10, 5))

        ep = range(1, self.length + 1)

        axes[0].plot(ep, self.train_cost, label="Training Cost")
        axes[0].plot(ep, self.val_cost, label="Validation Cost")
        axes[1].plot(ep, self.train_accuracy, label="Training Accuracy")
        axes[1].plot(ep, self.val_accuracy, label="Validation Accuracy")

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")

        for ax in axes:
            if self.title is not None:
                ax.set_title(self.title)

            ax.legend()
            ax.grid()

    def save(self, dirpath, params=None, postfix=None):
        with open(self._path(dirpath, params, postfix), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, dirpath, params=None, postfix=None):
        with open(cls._path(dirpath, params, postfix), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _path(dirpath, params=None, postfix=None):
        path = os.path.join(dirpath, 'history')

        if params is not None:
            for p, val in params:
                path += '_' + p + str(val).replace('.', '_')

        if postfix is not None:
            path += '_' + postfix

        return path + '.pickle'


class TrainHistoryRecurrent:
    def __init__(self, loss):
        self.loss = loss

    def visualize(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 5))

        ax.plot(range(1, len(self.loss) + 1), self.loss)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")

        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        ax.grid()

    def save(self, dirpath, postfix=None):
        with open(self._path(dirpath, postfix), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, dirpath, postfix=None):
        with open(cls._path(dirpath, postfix), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _path(dirpath, postfix=None):
        path = os.path.join(dirpath, 'history')

        if postfix is not None:
            path += '_' + postfix

        return path + '.pickle'
