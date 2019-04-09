import numpy as np

from network import Network


class RecurrentNetwork(Network):
    def __init__(self,
                 input_size,
                 hidden_state_size,
                 random_seed=None):

        super().__init__(random_seed)

        self.input_size = input_size

        self.b = self._rand_param((hidden_state_size, 1))
        self.c = self._rand_param((input_size, 1))
        self.U = self._rand_param((hidden_state_size, input_size))
        self.W = self._rand_param((hidden_state_size, hidden_state_size))
        self.V = self._rand_param((input_size, hidden_state_size))

    def params(self):
        return [self.b, self.c, self.U, self.W, self.V]

    def param_names(self):
        return ['b', 'c', 'U', 'W', 'V']

    def evaluate(self, ds):
        raise ValueError('TODO')

    def cost(self, ds, return_loss=False):
        raise ValueError('TODO')

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
        raise ValueError('TODO')
