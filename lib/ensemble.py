import numpy as np

from _visualize_performance import _visualize_performance


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
