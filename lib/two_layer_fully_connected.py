import numpy as np

from mlp_network import MLPNetwork


class TwoLayerFullyConnected(MLPNetwork):
    def __init__(self,
                 input_size,
                 hidden_nodes,
                 num_classes,
                 alpha=0,
                 weight_init='xavier',
                 dropout=None,
                 random_seed=None):

        super().__init__(random_seed)

        self.input_size = input_size
        self.hidden_nodes = hidden_nodes
        self.num_classes = num_classes

        self.alpha = alpha
        self.dropout = dropout

        self.W1 = self._rand_param((hidden_nodes, input_size), init=weight_init)
        self.W2 = self._rand_param((num_classes, hidden_nodes), init=weight_init)
        self.b1 = np.zeros((hidden_nodes, 1))
        self.b2 = np.zeros((num_classes, 1))

    @property
    def params(self):
        return [self.W1, self.W2, self.b1, self.b2]

    @params.setter
    def params(self, params):
        self.W1, self.W2, self.b1, self.b2 = params

    @property
    def param_names(self):
        return ['W1', 'W2', 'b1', 'b2']

    def evaluate(self, ds, return_H=False):
        H = self._relu(self.W1 @ ds.X + self.b1)

        if self.dropout is not None and self.dropout > 0:
            # use inverted dropout
            U = np.full_like(H, 1 / self.dropout)
            U[np.random.rand(*H.shape) < self.dropout] = 0

            H *= U

        P = self._softmax(self.W2 @ H + self.b2)

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
