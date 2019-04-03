import os
import pickle
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


NUM_GRAD_DELTA = 1e-6
ETA_DEFAULT = 0.01
N_BATCH_DEFAULT = 100
N_EPOCHS_DEFAULT = 1
N_DEAD_EPOCHS_MAX_DEFAULT = 1


class TrainHistory:
    def __init__(self):
        self.train_cost = []
        self.train_accuracy = []
        self.val_cost = []
        self.val_accuracy = []
        self.final_network = None

        self.length = 0

    def extend(self, network, ds_train, ds_val):
        self.train_cost.append(network.cost(ds_train))
        self.train_accuracy.append(network.accuracy(ds_train))
        self.val_cost.append(network.cost(ds_val))
        self.val_accuracy.append(network.accuracy(ds_val))

        self.length += 1

    def add_final_network(self, network):
        self.final_network = network

    def visualize(self, axes=None, title=None):
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
            if title is not None:
                ax.set_title(title)

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


class Network(ABC):
    @property
    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def evaluate(self, ds):
        pass

    @abstractmethod
    def cost(self, ds):
        pass

    @abstractmethod
    def gradients(self, ds, numerical=False, h=NUM_GRAD_DELTA):
        pass

    @abstractmethod
    def update(self, gradients, eta):
        pass

    def accuracy(self, ds):
        P = self.evaluate(ds)

        correct = np.count_nonzero(ds.y == P.argmax(axis=0))

        return correct / ds.n

    def train(self,
              ds_train,
              ds_val,
              eta=ETA_DEFAULT,
              n_batch=N_BATCH_DEFAULT,
              n_epochs=N_EPOCHS_DEFAULT,
              n_dead_epochs_max=N_DEAD_EPOCHS_MAX_DEFAULT,
              shuffle=False,
              stop_early=False,
              find_best_params=False,
              verbose=False):

        # keep track of loss and accuracy histories
        history = TrainHistory()

        # keep track of best parameters
        if stop_early:
            acc_best = 0
            dead_epochs = 0

        if find_best_params:
            params_best = None

        for ep in range(n_epochs):
            # optionally shuffle training data
            if shuffle:
                ds_train = ds_train.shuffle()

            n_batches = ds_train.n // n_batch

            for i in range(n_batches):
                # display progress
                if verbose:
                    fmt = f"epoch {ep + 1}/{n_epochs}, batch {i + 1}/{n_batches}"

                    if ep < n_epochs - 1 or i < (ds_train.n // n_batch) - 1:
                        print(fmt.ljust(80) + "\r", end='', flush=True)
                    else:
                        print(fmt.ljust(80), flush=True)

                # form batch
                i_start = i * n_batch
                i_end = (i + 1) * n_batch

                # update parameters
                gradients = self.gradients(ds_train.batch(i_start, i_end))
                self.update(gradients, eta)

            # extend history
            history.extend(self, ds_train, ds_val)

            acc_last = history.val_accuracy[-1]

            if stop_early:
                if acc_last > acc_best:
                    acc_best = acc_last
                    dead_epochs = 0
                else:
                    dead_epochs += 1

                    if dead_epochs >= n_dead_epochs_max:
                        break

            if find_best_params:
                if acc_last > acc_best:
                    params_best = [p.copy() for p in self.params]

        if find_best_params:
            self.params = params_best

        history.add_final_network(self)

        return history

    def visualize_performance(self, ds, ax=None, title=None):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 8))

        acc = self.accuracy(ds)

        y_pred = np.argmax(self.evaluate(ds), axis=0)

        df = pd.DataFrame(confusion_matrix(np.squeeze(ds.y), y_pred),
                          index=ds.labels,
                          columns=ds.labels)

        hm = sns.heatmap(
            df, cbar=False, annot=True, fmt='d', cmap='Blues', ax=ax)

        xlabels = hm.xaxis.get_ticklabels()
        hm.xaxis.set_ticklabels(xlabels, rotation=45, ha='right')

        if title is not None:
            fmt = title + ", Total Accuracy is {:.3f}"
        else:
            fmt = "Total Accuracy is {:.3f}"

        ax.set_title(fmt.format(acc))

        plt.tight_layout()

    def visualize_weights(self, axes=None):
        if not hasattr(self, 'W'):
            raise ValueError(
                "weight visualization not implemented for this network type")

        if axes is None:
            fig, axes = plt.subplots(2, self.W.shape[0] // 2, figsize=(8, 4))
            fig.subplots_adjust(hspace=0.1, wspace=0.1)

        for w, ax in zip(self.W, axes.flatten()):
            img = ((w - w.min()) / (w.max() - w.min()))

            ax.imshow(img.reshape(3, 32, 32).transpose(1, 2, 0))

            ax.tick_params(axis='both',
                           which='both',
                           bottom=False,
                           top=False,
                           left=False,
                           right=False,
                           labelbottom=False,
                           labelleft=False)


class SingleLayerFullyConnected(Network):
    PARAM_STD = 0.01
    PARAM_DTYPE = np.float64

    def __init__(self, input_size, num_classes, alpha=0):
        self.input_size = input_size
        self.num_classes = num_classes

        self.W = self._rand_param((num_classes, input_size))
        self.b = self._rand_param((num_classes, 1))
        self.alpha = alpha

    def _rand_param(self, shape):
        return self.PARAM_STD * np.random.randn(*shape).astype(self.PARAM_DTYPE)

    @property
    def params(self):
        return [self.W, self.b]

    @params.setter
    def params(self, params):
        self.W, self.b = params

    def evaluate(self, ds):
        s = self.W @ ds.X + self.b

        P = np.exp(s)
        P /= P.sum(axis=0)

        return P

    def cost(self, ds):
        P = self.evaluate(ds)
        py = P[ds.y, range(ds.n)]

        loss = -np.mean(np.log(py))
        reg = self.alpha * np.sum(self.W**2)

        return loss + reg

    def gradients(self, ds, numerical=False, h=NUM_GRAD_DELTA):
        if numerical:
            grad_W = np.zeros_like(self.W)
            grad_b = np.zeros((self.num_classes, 1), dtype=self.W.dtype)

            c1 = self.cost(ds)

            for i in range(self.num_classes):
                self.b[i] += h
                c2 = self.cost(ds)
                grad_b[i] = (c2 - c1) / h
                self.b[i] -= h

            for i in range(self.num_classes):
                for j in range(self.input_size):
                    self.W[i, j] += h
                    c2 = self.cost(ds)
                    grad_W[i, j] = (c2 - c1) / h
                    self.W[i, j] -= h
        else:
            P = self.evaluate(ds)

            G = -(ds.Y - P)

            grad_W = 1 / ds.n * G @ ds.X.T + 2 * self.alpha * self.W
            grad_b = 1 / ds.n * G.sum(axis=1, keepdims=True)

        return [grad_W, grad_b]

    def update(self, gradients, eta):
        self.W -= eta * gradients[0]
        self.b -= eta * gradients[1]
