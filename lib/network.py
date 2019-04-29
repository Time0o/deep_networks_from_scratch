from abc import ABC, abstractmethod

import numpy as np


PARAM_DTYPE = np.float64
PARAM_STD_DEFAULT = 0.01
NUM_GRAD_DELTA = 1e-6


class Network(ABC):
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

    def gradients(self, ds, numerical=False, h=NUM_GRAD_DELTA):
        if numerical:
            return self._gradients_numerical(ds, h)
        else:
            return self._gradients(ds)

    def update(self, gradients, eta):
        for i, param in enumerate(self.params):
            param -= eta * gradients[i]

    def _rand_param(self, shape, init='standard', std=PARAM_STD_DEFAULT):
        if init == 'standard':
            f = std
        elif init == 'xavier':
            f = 1 / np.sqrt(shape[1])
        elif init == 'he':
            f = np.sqrt(2 / shape[1])
        else:
            raise ValueError("invalid parameter initializer")

        return f * np.random.randn(*shape).astype(PARAM_DTYPE)

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
