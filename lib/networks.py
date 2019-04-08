import math
import os
import pickle
from abc import ABC, abstractmethod
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


NUM_GRAD_DELTA = 1e-6
ETA_DEFAULT = 0.01
ETA_MIN_DEFAULT = 1e-5
ETA_MAX_DEFAULT = 1e-1
ETA_SS_DEFAULT = 500
N_BATCH_DEFAULT = 100
N_EPOCHS_DEFAULT = 1
N_CYCLES_DEFAULT = 1
HISTORY_PER_CYCLE_DEFAULT = 10
N_DEAD_EPOCHS_MAX_DEFAULT = 1
BATCHNORM_ALPHA_DEFAULT = 0.9


def _visualize_performance(ds, acc, y_pred, ax=None, title=None):
     if ax is None:
         _, ax = plt.subplots(1, 1, figsize=(8, 8))

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


class Network(ABC):
    PARAM_DTYPE = np.float64
    PARAM_STD_DEFAULT = 0.01

    @abstractmethod
    def __init__(self, random_seed):
        if random_seed is not None:
            np.random.seed(random_seed)

    @property
    @abstractmethod
    def params(self):
        pass

    @property
    @abstractmethod
    def param_names(self):
        pass

    @abstractmethod
    def evaluate(self, ds):
        pass

    @abstractmethod
    def cost(self, ds, return_loss=False):
        pass

    def gradients(self, ds, numerical=False, h=NUM_GRAD_DELTA):
        if numerical:
            return self._gradients_numerical(ds, h)
        else:
            return self._gradients(ds)

    def update(self, gradients, eta):
        for i, param in enumerate(self.params):
            param -= eta * gradients[i]

    def predict(self, ds):
        return self.evaluate(ds).argmax(axis=0)

    def accuracy(self, ds):
        correct = np.count_nonzero(ds.y == self.predict(ds))

        return correct / ds.n

    def train(self,
              ds_train,
              ds_val,
              eta=ETA_DEFAULT,
              eta_decay_factor=None,
              n_batch=N_BATCH_DEFAULT,
              n_epochs=N_EPOCHS_DEFAULT,
              n_dead_epochs_max=N_DEAD_EPOCHS_MAX_DEFAULT,
              shuffle=False,
              stop_early=False,
              stop_early_metric='loss',
              stop_early_find_best_params=False,
              stop_early_best_params_metric='acc',
              verbose=False):

        # keep track of loss and accuracy histories
        history = TrainHistory()

        # keep track of best parameters
        if stop_early:
            dead_epochs = 0

            loss_best = math.inf
            acc_best = 0

            if stop_early_find_best_params:
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

            if stop_early:
                loss_last = history.val_loss[-1]
                acc_last = history.val_accuracy[-1]

                loss_improved = loss_last < loss_best
                acc_improved = acc_last > acc_best

                if (stop_early_metric == 'loss' and loss_improved) or \
                   (stop_early_metric == 'acc' and acc_improved):

                    dead_epochs = 0

                else:
                    dead_epochs += 1

                    if dead_epochs >= n_dead_epochs_max:
                        break

                if stop_early_find_best_params:
                    if (stop_early_best_params_metric == 'loss' and loss_improved) or \
                       (stop_early_best_params_metric == 'acc' and acc_improved):

                        params_best = [p.copy() for p in self.params]

                if loss_last < loss_best:
                    loss_best = loss_last

                if acc_last > acc_best:
                    acc_best = acc_last

            # decay learning rate
            if eta_decay_factor is not None:
                eta *= eta_decay_factor

        if stop_early and stop_early_find_best_params:
            self.params = params_best

        history.add_final_network(self)

        return history

    def train_cyclic(self,
                     ds_train,
                     ds_val,
                     eta_min=ETA_MIN_DEFAULT,
                     eta_max=ETA_MAX_DEFAULT,
                     eta_ss=ETA_SS_DEFAULT,
                     n_batch=N_BATCH_DEFAULT,
                     n_cycles=N_CYCLES_DEFAULT,
                     history_per_cycle=HISTORY_PER_CYCLE_DEFAULT,
                     shuffle=False,
                     verbose=False):

        # keep track of loss and accuracy histories
        history = TrainHistory(store_learning_rate=True)

        # update loop
        n_updates = 2 * eta_ss * n_cycles
        update = 0
        done = False

        while not done:
            # optionally shuffle training data
            if shuffle:
                ds_train = ds_train.shuffle()

            n_batches = ds_train.n // n_batch

            for i in range(n_batches):
                # display progress
                if verbose:
                    fmt = "update {}/{}"
                    fmt = fmt.format(update + 1, n_updates)

                    if update == n_updates - 1:
                        print(fmt.ljust(80), flush=True)
                    else:
                        print(fmt.ljust(80) + "\r", end='', flush=True)

                # form batch
                i_start = i * n_batch
                i_end = (i + 1) * n_batch

                # determine current learning rate
                t = update % (2 * eta_ss)

                if t <= eta_ss:
                    eta = eta_min + t / eta_ss * (eta_max - eta_min)
                else:
                    eta = eta_max - (t - eta_ss) / eta_ss * (eta_max - eta_min)

                history.add_learning_rate(eta)

                # update parameters
                gradients = self.gradients(ds_train.batch(i_start, i_end))

                self.update(gradients, eta)

                # extend history
                if update % (2 * eta_ss // history_per_cycle) == 0:
                    history.extend(self, ds_train, ds_val)

                update += 1
                if update == n_updates:
                    done = True
                    break

        history.add_final_network(self)

        return history

    def visualize_performance(self, ds, ax=None, title=None):
        acc = self.accuracy(ds)
        y_pred = self.predict(ds)

        _visualize_performance(ds, acc, y_pred, ax=ax, title=title)

    def visualize_weights(self, axes=None):
        if not hasattr(self, 'W'):
            raise ValueError(
                "weight visualization not implemented for this network type")

        if axes is None:
            fig, axes = plt.subplots(2, self.W.shape[0] // 2, figsize=(8, 4))
            fig.subplots_adjust(hspace=0.1, wspace=0.1)

        for w, ax in zip(self.W, axes.flatten()):
            w = w[:3072]

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

    def _rand_param(self, shape, init='standard'):
        if init == 'standard':
            f = self.PARAM_STD_DEFAULT
        elif init == 'xavier':
            f = 1 / np.sqrt(shape[1])
        elif init == 'he':
            f = np.sqrt(2 / shape[1])
        else:
            raise ValueError("invalid parameter initializer")

        return f * np.random.randn(*shape).astype(self.PARAM_DTYPE)

    def _gradients_numerical(self, ds, h):
        grads = []

        for param in self.params:
            grad = np.zeros_like(param)

            c1 = self.cost(ds)

            for i in range(grad.shape[0]):
                for j in range(grad.shape[1]):
                    param[i, j] += h
                    c2 = self.cost(ds)
                    grad[i, j] = (c2 - c1) / h
                    param[i, j] -= h

            grads.append(grad)

        return grads

    @abstractmethod
    def _gradients(self, ds):
        pass


class SingleLayerFullyConnected(Network):
    def __init__(self,
                 input_size,
                 num_classes,
                 alpha=0,
                 loss='cross_entropy',
                 weight_init='standard',
                 random_seed=None):

        super().__init__(random_seed)

        if loss not in ['cross_entropy', 'svm']:
            raise ValueError("'loss' must be either 'cross_entropy' or 'svm'")

        self.input_size = input_size
        self.num_classes = num_classes

        self.alpha = alpha
        self._svm_loss = loss == 'svm'

        self.W = self._rand_param((num_classes, input_size), init=weight_init)

        if not self._svm_loss:
            self.b = self._rand_param((num_classes, 1))

    @property
    def param_names(self):
        if self._svm_loss:
            return ['W']
        else:
            return ['W', 'b']

    @property
    def params(self):
        if self._svm_loss:
            return [self.W]
        else:
            return [self.W, self.b]

    @params.setter
    def params(self, params):
        if self._svm_loss:
            self.W = params[0]
        else:
            self.W, self.b = params

    def evaluate(self, ds):
        if self._svm_loss:
            return self.W @ ds.X
        else:
            s = self.W @ ds.X + self.b

            P = np.exp(s)
            P /= P.sum(axis=0)

            return P

    def cost(self, ds, return_loss=False):
        if self._svm_loss:
            delta = self._svm_delta(ds)

            loss = delta.sum() / ds.n
            reg = self.alpha * np.sum(self.W**2)
        else:
            P = self.evaluate(ds)
            py = P[ds.y, range(ds.n)]

            loss = -np.mean(np.log(py))
            reg = self.alpha * np.sum(self.W**2)

        cost = loss + reg

        if return_loss:
            return loss, cost
        else:
            return cost

    def _svm_delta(self, ds):
        s = self.evaluate(ds)

        return np.maximum(0, (s - np.sum(s * ds.Y, axis=0) + 1) * (1 - ds.Y))

    def _gradients(self, ds):
        if self._svm_loss:
            delta = self._svm_delta(ds)

            ind = delta
            ind[delta > 0] = 1;
            ind[np.argmax(ds.Y, axis=0),
                np.arange(ds.n)] = -np.sum(ind, axis=0)

            grad_W = ind @ ds.X.T / ds.n + 2 * self.alpha * self.W

            return [grad_W]
        else:
            P = self.evaluate(ds)

            G = -(ds.Y - P)

            grad_W = 1 / ds.n * G @ ds.X.T + 2 * self.alpha * self.W
            grad_b = 1 / ds.n * G.sum(axis=1, keepdims=True)

            return [grad_W, grad_b]


class TwoLayerFullyConnected(Network):
    def __init__(self,
                 input_size,
                 hidden_nodes,
                 num_classes,
                 alpha=0,
                 weight_init='he',
                 random_seed=None):

        super().__init__(random_seed)

        self.input_size = input_size
        self.hidden_nodes = hidden_nodes
        self.num_classes = num_classes

        self.alpha = alpha

        self.W1 = self._rand_param((hidden_nodes, input_size), init=weight_init)
        self.W2 = self._rand_param((num_classes, hidden_nodes), init=weight_init)
        self.b1 = np.zeros((hidden_nodes, 1))
        self.b2 = np.zeros((num_classes, 1))

    @property
    def params(self):
        return [self.W1, self.W2, self.b1, self.b2]

    @property
    def param_names(self):
        return ['W1', 'W2', 'b1', 'b2']

    def evaluate(self, ds, return_H=False):
        H = self.W1 @ ds.X + self.b1
        H[H < 0] = 0

        S = self.W2 @ H + self.b2

        P = np.exp(S)
        P /= P.sum(axis=0)

        if return_H:
            return H, P
        else:
            return P

    def cost(self, ds, return_loss=False):
        P = self.evaluate(ds)
        py = P[ds.y, range(ds.n)]

        loss = -np.log(py).sum() / ds.n
        reg = self.alpha * np.sum([np.sum(self.W1**2) + np.sum(self.W2**2)])

        cost = loss + reg

        if return_loss:
            return loss, cost
        else:
            return cost

    def _gradients(self, ds):
        H, P = self.evaluate(ds, return_H=True)

        G = -(ds.Y - P)

        grad_W2 = 1 / ds.n * G @ H.T + 2 * self.alpha * self.W2
        grad_b2 = 1 / ds.n * np.sum(G, axis=1, keepdims=True)

        G = self.W2.T @ G
        G = G * (H > 0)

        grad_W1 = 1 / ds.n * G @ ds.X.T + 2 * self.alpha * self.W1
        grad_b1 = 1 / ds.n * np.sum(G, axis=1, keepdims=True)

        return [grad_W1, grad_W2, grad_b1, grad_b2]


class MultiLayerFullyConnected(Network):
    def __init__(self,
                 input_size,
                 hidden_nodes,
                 num_classes,
                 alpha=0,
                 weight_init='he',
                 batchnorm=False,
                 batchnorm_alpha=BATCHNORM_ALPHA_DEFAULT,
                 random_seed=None):

        super().__init__(random_seed)

        # network dimensions
        self.input_size = input_size
        self.hidden_nodes = hidden_nodes
        self.num_classes = num_classes

        # regularization
        self.alpha = alpha
        self.batchnorm = batchnorm
        self.batchnorm_alpha = batchnorm_alpha

        # initialize parameters
        self.Ws = []
        self.bs = []

        for d1, d2 in zip([input_size] + hidden_nodes,
                          hidden_nodes + [num_classes]):

            self.Ws.append(self._rand_param((d2, d1), init=weight_init))
            self.bs.append(np.zeros((d2, 1)))

        if batchnorm:
            self.gamma = [np.ones((d, 1)) for d in hidden_nodes]
            self.beta = [np.zeros((d, 1)) for d in hidden_nodes]

            self.mu_train = None
            self.var_train = None

    @property
    def params(self):
        params = self.Ws + self.bs

        if self.batchnorm:
            params += self.gamma + self.beta

        return params

    @property
    def param_names(self):
        def names(params, fmt):
            return [f'{fmt}{i+1}' for i in range(len(params))]

        param_names = names(self.Ws, 'W') + names(self.bs, 'b')

        if self.batchnorm:
            param_names += names(self.gamma, 'gamma') + names(self.beta, 'beta')

        return param_names

    def evaluate(self, ds, training=False):
        if self.batchnorm and training:
            raw = []
            normalized = []
            mu = []
            var = []

        X = ds.X
        activations = [X]

        for i in range(len(self.hidden_nodes)):
            X = self.Ws[i] @ X + self.bs[i]

            if self.batchnorm:
                # batchnorm
                if training:
                    m = np.mean(X, axis=1, keepdims=True)
                    v = np.var(X, axis=1, keepdims=True)

                    raw.append(X)
                    mu.append(m)
                    var.append(v)
                else:
                    if self.mu_train is None:
                        m = np.mean(X, axis=1, keepdims=True)
                    else:
                        m = self.mu_train[i]

                    if self.var_train is None:
                        v = np.var(X, axis=1, keepdims=True)
                    else:
                        v = self.var_train[i]

                X = np.diag((np.squeeze(v) + np.spacing(1))**-0.5) @ (X - m)

                if training:
                    normalized.append(X)

                # scale and shift
                X = self.gamma[i] * X + self.beta[i]

            X[X < 0] = 0

            if training:
                activations.append(X)

        S = self.Ws[-1] @ X + self.bs[-1]

        P = np.exp(S)
        P /= P.sum(axis=0)

        if training:
            if self.batchnorm:
                self.last_mu = mu
                self.last_var = var
                return raw, normalized, activations, mu, var, P
            else:
                return activations, P
        else:
            return P

    def cost(self, ds, return_loss=False):
        P = self.evaluate(ds)
        py = P[ds.y, range(ds.n)]

        loss = -np.log(py).sum() / ds.n
        reg = self.alpha * np.sum([np.sum(W**2) for W in self.Ws])

        cost = loss + reg

        if return_loss:
            return loss, cost
        else:
            return cost

    def update(self, gradients, eta):
        super().update(gradients, eta)

        if not self.batchnorm:
            return

        alpha = self.batchnorm_alpha

        if self.mu_train is None:
            self.mu_train = self.last_mu
        else:
            self.mu_train = [
                alpha * mt + (1 - alpha) * m
                for mt, m in zip(self.mu_train, self.last_mu)
            ]

        if self.var_train is None:
            self.var_train = self.last_var
        else:
            self.var_train = [
                alpha * vt + (1 - alpha) * v
                for vt, v in zip(self.var_train, self.last_var)
            ]

    def _gradients(self, ds):
        if self.batchnorm:
            raw, normalized, activations, mu, var, P = \
                self.evaluate(ds, training=True)

            grads_W = []
            grads_b = []
            grads_gamma = []
            grads_beta = []

            G = -(ds.Y - P)

            # compute weight and bias gradients
            grad_W, grad_b = self._gradients_param(G, ds.n, activations, -1)
            grads_W.append(grad_W)
            grads_b.append(grad_b)

            # propagate G to previous layer
            G = self._propagate_layer(G, activations, -1)

            for i in range(len(self.hidden_nodes) - 1, -1, -1):
                # compute scale and shift gradients
                grad_gamma, grad_beta = self._gradients_ss(G, ds.n, normalized, i)
                grads_gamma.append(grad_gamma)
                grads_beta.append(grad_beta)

                # propagate G through scale and shift
                G = self._propagate_ss(G, i)

                # propagate G through batchnorm
                G = self._propagate_batchnorm(G, ds.n, raw, mu, var, i)

                # compute weight and bias gradients
                grad_W, grad_b = self._gradients_param(G, ds.n, activations, i)
                grads_W.append(grad_W)
                grads_b.append(grad_b)

                # propagate G to previous layer
                if i > 0:
                    G = self._propagate_layer(G, activations, i)

            return self._join_gradients(
                [grads_W, grads_b, grads_gamma, grads_beta])
        else:
            activations, P = self.evaluate(ds, training=True)

            G = -(ds.Y - P)

            grads_W = []
            grads_b = []

            for i in range(len(self.hidden_nodes), -1, -1):
                grad_W, grad_b = self._gradients_param(G, ds.n, activations, i)

                grads_W.append(grad_W)
                grads_b.append(grad_b)

                if i > 0:
                    G = self._propagate_layer(G, activations, i)

            return self._join_gradients([grads_W, grads_b])

    def _propagate_layer(self, G, activations, i):
        G = self.Ws[i].T @ G
        G = G * (activations[i] > 0)

        return G

    def _propagate_ss(self, G, i):
        return G * self.gamma[i]

    def _propagate_batchnorm(self, G, n, raw, mu, var, i):
        sigma1 = (var[i] + np.spacing(1))**-0.5
        sigma2 = (var[i] + np.spacing(1))**-1.5

        G1 = G * sigma1
        G2 = G * sigma2

        D = raw[i] - mu[i]
        c = np.sum(G2 * D, axis=1, keepdims=True)

        return G1 - G1.sum(axis=1, keepdims=True) / n - (D * c) / n

    def _gradients_param(self, G, n, activations, i):
        grad_W = (G @ activations[i].T) / n + 2 * self.alpha * self.Ws[i]
        grad_b = G.sum(axis=1, keepdims=True) / n

        return grad_W, grad_b

    def _gradients_ss(self, G, n, normalized, i):
        grad_gamma = np.sum(G * normalized[i], axis=1, keepdims=True) / n
        grad_beta = G.sum(axis=1, keepdims=True) / n

        return grad_gamma, grad_beta

    @staticmethod
    def _join_gradients(grads):
        return list(chain(*[reversed(g) for g in grads]))


class EnsembleClassifier:
    def __init__(self, networks):
        self.networks = networks

    def predict(self, ds):
        y_pred = np.empty((len(self.networks), ds.n), dtype=int)

        for i, network in enumerate(self.networks):
            y_pred[i, :] = network.evaluate(ds).argmax(axis=0)

        y_pred_ensemble = np.empty(ds.n, dtype=int)
        for i in range(ds.n):
            bc = np.bincount(y_pred[:, i])
            pred_max = np.flatnonzero(bc == bc.max())
            y_pred_ensemble[i] = np.random.choice(pred_max)

        return y_pred_ensemble

    def accuracy(self, ds):
        correct = np.count_nonzero(ds.y == self.predict(ds))

        return correct / ds.n

    def visualize_performance(self, ds, ax=None, title=None):
        acc = self.accuracy(ds)
        y_pred = self.predict(ds)

        _visualize_performance(ds, acc, y_pred, ax=ax, title=title)
