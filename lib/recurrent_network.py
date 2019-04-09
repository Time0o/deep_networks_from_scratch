import numpy as np

from network import Network


class RecurrentNetwork(Network):
    def __init__(self,
                 input_size,
                 hidden_state_size,
                 random_seed=None):

        super().__init__(random_seed)

        self.input_size = input_size
        self.hidden_state_size = hidden_state_size

        self.b = self._rand_param((hidden_state_size, 1))
        self.c = self._rand_param((input_size, 1))
        self.U = self._rand_param((hidden_state_size, input_size))
        self.W = self._rand_param((hidden_state_size, hidden_state_size))
        self.V = self._rand_param((input_size, hidden_state_size))

    @property
    def params(self):
        return [self.b, self.c, self.U, self.W, self.V]

    @property
    def param_names(self):
        return ['b', 'c', 'U', 'W', 'V']

    def evaluate(self, ds, return_intermediate=False, return_loss=False):
        P = np.empty((self.input_size, ds.n))
        A = np.empty((self.hidden_state_size, ds.n))
        H = np.empty((self.hidden_state_size, ds.n))

        h = np.zeros((self.hidden_state_size, 1))

        loss = 0
        for i, (x, y) in enumerate(zip(ds.X.T, ds.Y.T)):
            x = x[:, np.newaxis]
            y = y[:, np.newaxis]

            a = self.W @ h + self.U @ x + self.b
            A[:, i] = np.squeeze(a)

            h = np.tanh(a)
            H[:, i] = np.squeeze(h)

            o = self.V @ h + self.c

            p = np.exp(o)
            p /= p.sum(axis=0)
            P[:, i] = np.squeeze(p)

            loss += np.log(y.T @ p)

        ret = [P]

        if return_intermediate:
            ret += [A, H]

        if return_loss:
            ret.append(loss)

        return ret[0] if len(ret) == 1 else ret

    def cost(self, ds):
        _, loss = self.evaluate(ds, return_loss=True)
        return loss

    def synthesize(self, x_init, h_init, length):
        Y = np.empty((self.input_size, length))

        h = h_init
        x = x_init

        for i in range(length):
            a = self.W @ h + self.U @ x + self.b
            h = np.tanh(a)
            o = self.V @ h + self.c

            p = np.exp(o)
            p /= p.sum(axis=0)

            x = np.random.multinomial(1, np.squeeze(p))[:, np.newaxis]

            Y[:, i] = np.squeeze(x)

        return Y

    def _gradients(self, ds):
        P, A, H = self.evaluate(ds, return_intermediate=True)

        grad_h = np.zeros((ds.n, self.hidden_state_size), dtype=self.PARAM_DTYPE)
        grad_a = np.zeros((ds.n, self.hidden_state_size), dtype=self.PARAM_DTYPE)

        grad_o = -(ds.Y - P).T
        grad_c = grad_o.sum(axis=0)[:, np.newaxis]
        grad_V = grad_o.T @ H.T

        grad_h[ds.n - 1, :] = \
            grad_o[ds.n - 1, :] @ self.V

        grad_a[ds.n - 1, :] = \
            grad_h[ds.n - 1, :] @ np.diag(1 - np.tanh(A[:, ds.n - 1])**2)

        for t in range(ds.n - 2, -1, -1):
            grad_h[t, :] = grad_o[t, :] @ self.V + grad_a[t + 1, :] @ self.W
            grad_a[t, :] = grad_h[t, :] @ np.diag(1 - np.tanh(A[:, t])**2)

        grad_b = grad_a.sum(axis=0)[:, np.newaxis]
        grad_W = grad_a.T @ H.T
        grad_U = grad_a.T @ ds.X.T

        return [grad_b, grad_c, grad_U, grad_W, grad_V]
