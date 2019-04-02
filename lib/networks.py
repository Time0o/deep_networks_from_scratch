from abc import ABC, abstractmethod

import numpy as np


NUM_GRAD_DELTA = 1e-6


class Network(ABC):
    @abstractmethod
    def evaluate(self, ds):
        pass

    @abstractmethod
    def cost(self, ds):
        pass

    @abstractmethod
    def gradients(self, ds, numerical=False, h=NUM_GRAD_DELTA):
        pass

    def accuracy(self, ds):
        P = self.evaluate(ds)

        correct = np.count_nonzero(ds.y == P.argmax(axis=0))

        return correct / ds.n


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

        return grad_W, grad_b
