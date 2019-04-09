import numpy as np

from network import Network


class Recurrent(Network):
    def __init__(self,
                 input_size,
                 hidden_state_size,
                 random_seed=None):

        super().__init__(random_seed)

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

    def _gradients(self, ds):
        raise ValueError('TODO')
