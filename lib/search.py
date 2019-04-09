import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


class SearchParam:
    def __init__(self,
                 name,
                 min,
                 max,
                 n,
                 dtype=np.float64,
                 scale='linear'):

        self.name = name
        self.min = min
        self.max = max
        self.n = n
        self.dtype = dtype
        self.scale = scale

    def grid_samples(self):
        samples = np.linspace(
            self.min, self.max, self.n, dtype=self.dtype)

        return self._transform_samples(samples)

    def random_samples(self):
        samples = np.random.uniform(
            self.min, self.max, self.n).astype(self.dtype)

        return self._transform_samples(samples)

    def _transform_samples(self, samples):
        if self.scale == 'linear':
            return samples
        elif self.scale == 'log':
            return 10**samples


class SearchResult:
    def __init__(self, params, cost_min, loss_min, acc_max):
        self.params = params
        self.cost_min = cost_min
        self.loss_min = loss_min
        self.acc_max = acc_max


class SearchResultSeries:
    def __init__(self, results):
        self.results = results

    def optimum(self):
        params = [res.params for res in self.results]
        accs = [res.acc_max for res in self.results]

        return params[np.argmax(accs)]

    def summarize(self, top=5):
        params = [res.params for res in self.results]
        accs = [res.acc_max for res in self.results]

        print("best accuracies:")

        for acc, ps in sorted(zip(accs, params), reverse=True)[:top]:
            param_strings = []

            for name, val in ps.items():
                if isinstance(val, float):
                    fmt = '{} = {:1.3e}'
                else:
                    fmt = '{} = {}'

                param_strings.append(fmt.format(name, val))

            print("accuracy = {:.4f} ({})".format(
                acc, ', '.join(param_strings)))

    def visualize(self, param, param_=None):
        values = [res.params[param.name] for res in self.results]
        costs = [res.cost_min for res in self.results]
        accs = [res.acc_max for res in self.results]

        if param_ is None:
            _, (ax_cost, ax_acc) = plt.subplots(1, 2, figsize=(10, 5))

            ax_cost.stem(values, costs)
            ax_cost.set_ylabel("Cost")

            ax_acc.stem(values, accs)
            ax_acc.set_ylabel("Accuracy")

            for ax in ax_cost, ax_acc:
                ax.set_xlabel(param.name)

                if param.scale == 'log':
                    ax.set_xscale('log')

            plt.tight_layout()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            cax = fig.add_axes()

            values_ = [res.params[param_.name] for res in results]

            grid = np.linspace(min(values), max(values), 100)
            grid_ = np.linspace(min(values_), max(values_), 100)

            grid, grid_ = np.meshgrid(grid, grid_)

            gridd = griddata((values, values_), accs, (grid, grid_))

            ax.contour(grid, grid_, gridd, colors='k')
            contour = ax.contourf(grid, grid_, gridd, cmap=cm.jet)
            ax.scatter(values, values_, marker='x', color='k')

            fig.colorbar(contour, cax=cax)

            ax.set_xlabel(param.name)
            ax.set_ylabel(param_.name)

            # TODO: log

    def save(self, dirpath, postfix=None):
        with open(self._path(dirpath, postfix), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, dirpath, postfix=None):
        with open(cls._path(dirpath, postfix), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _path(dirpath, postfix=None):
        path = os.path.join(dirpath, 'search')

        if postfix is not None:
            path += '_' + postfix

        return path + '.pickle'


def search(data,
           data_val,
           params,
           train_function,
           random=True,
           random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    param_names = [p.name for p in params]

    if random:
        param_value_iter = zip(*[p.random_samples() for p in params])
    else:
        param_value_iter = product(*[p.grid_samples() for p in params])

    results = []

    for param_values in param_value_iter:
        param_args = {
            name: val
            for name, val in zip(param_names, param_values)
        }

        history = train_function(param_args)

        results.append(SearchResult(param_args,
                                    np.min(history.val_cost),
                                    np.min(history.val_loss),
                                    np.max(history.val_accuracy)))

    return SearchResultSeries(results)
