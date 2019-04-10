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

        # hidden state
        self.h = np.zeros((hidden_state_size, 1))

        # weights
        self.U = self._rand_param((hidden_state_size, input_size))
        self.V = self._rand_param((input_size, hidden_state_size))
        self.W = self._rand_param((hidden_state_size, hidden_state_size))

        # biases
        self.b = np.zeros((hidden_state_size, 1))
        self.c = np.zeros((input_size, 1))

    @property
    def params(self):
        return [self.U, self.V, self.W, self.b, self.c]

    @property
    def param_names(self):
        return ['U', 'V', 'W', 'b', 'c']

    def evaluate(self, ds, return_intermediate=False, return_loss=False):
        # allocate result matrix
        P = np.empty((self.input_size, ds.n))

        # allocate intermediate result matrices
        if return_intermediate:
            H = np.empty((self.hidden_state_size, ds.n + 1))

        # initialize hidden state
        h = self.h.copy()

        if return_intermediate:
            H[:, 0] = np.squeeze(h)

        # initialize loss
        if return_loss:
            loss = 0

        for i, (x, y) in enumerate(zip(ds.X.T, ds.Y.T)):
            x = x[:, np.newaxis]
            y = y[:, np.newaxis]

            a = self.W @ h + self.U @ x + self.b
            h = np.tanh(a)

            if return_intermediate:
                H[:, i + 1] = np.squeeze(h)

            o = self.V @ h + self.c

            p = np.exp(o)
            p /= p.sum(axis=0)
            P[:, i] = np.squeeze(p)

            if return_loss:
                loss -= np.log((y.T @ p).item())

        # cobble together return value
        ret = [P]

        if return_intermediate:
            ret.append(H)

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
        P, H = self.evaluate(ds, return_intermediate=True)

        grad_o = -(ds.Y - P).T
        grad_a = np.zeros((self.hidden_state_size, 1))

        grad_U = np.zeros_like(self.U)
        grad_V = grad_o.T @ H[:, 1:].T
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        grad_c = grad_o.sum(axis=0)[:, np.newaxis]

        for t in range(ds.n - 1, -1, -1):
            grad_h = (grad_o[np.newaxis, t, :] @ self.V + grad_a.T @ self.W).T
            grad_a = grad_h * (1 - H[:, t + 1, np.newaxis]**2)

            grad_W += np.outer(grad_a, H[:, t, np.newaxis])
            grad_U += np.outer(grad_a, ds.X[:, t, np.newaxis])
            grad_b += grad_a

        return [grad_U, grad_V, grad_W, grad_b, grad_c]
