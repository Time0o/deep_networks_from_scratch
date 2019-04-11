import matplotlib.pyplot as plt
import numpy as np

from mlp_network import MLPNetwork


class SingleLayerFullyConnected(MLPNetwork):
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

    def visualize_weights(self, axes=None):
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
