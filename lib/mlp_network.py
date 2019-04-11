import math
import numpy as np
from abc import abstractmethod

from _visualize_performance import _visualize_performance
from history import TrainHistory
from network import Network


ETA_DEFAULT = 0.01
ETA_MIN_DEFAULT = 1e-5
ETA_MAX_DEFAULT = 1e-1
ETA_SS_DEFAULT = 500
N_BATCH_DEFAULT = 100
N_EPOCHS_DEFAULT = 1
N_CYCLES_DEFAULT = 1
HISTORY_PER_CYCLE_DEFAULT = 10
N_DEAD_EPOCHS_MAX_DEFAULT = 1


class MLPNetwork(Network):
    @abstractmethod
    def cost(self, ds):
        pass

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
