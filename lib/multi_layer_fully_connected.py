from itertools import chain

import numpy as np

from mlp_network import MLPNetwork


BATCHNORM_ALPHA_DEFAULT = 0.9


class MultiLayerFullyConnected(MLPNetwork):
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

    @params.setter
    def params(self, params):
        self.Ws, self.bs = params[:2]

        if self.batchnorm:
            self.gamma, self.beta = params[2:]

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
