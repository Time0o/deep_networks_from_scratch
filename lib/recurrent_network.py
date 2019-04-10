from textwrap import wrap

import numpy as np

from history import TrainHistoryRecurrent
from network import Network


ETA_DEFAULT = 0.1
N_UPDATES_DEFAULT = 100000


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

        # adagrad
        self.adamem = [np.zeros_like(param) for param in self.params]

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

    def synthesize(self, x_init, length):
        Y = np.empty((self.input_size, length))

        h = self.h.copy()

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

    def update(self, grads, eta):
        for param, grad, mem in zip(self.params, grads, self.adamem):
            mem += grad * grad
            param -= eta * grad / np.sqrt(mem + np.spacing(1))

    def predict(self, ds):
        raise ValueError("not implemented")

    def accuracy(self, ds):
        raise ValueError("not implemented")

    def train(self,
              text,
              sequence_length,
              eta=ETA_DEFAULT,
              n_updates=N_UPDATES_DEFAULT,
              loss_smoothing_factor=0.999,
              verbose=False,
              verbose_show_loss=True,
              verbose_show_loss_frequency=1000,
              verbose_show_samples=False,
              verbose_show_samples_frequency=10000,
              verbose_show_samples_length=200):

        loss_smooth = []

        update = 0
        while update <= n_updates:
            for e in range(0, len(text.text) - sequence_length - 1, sequence_length):
                batch = text.sequence(beg=e,
                                      end=e + sequence_length,
                                      rep='indices_one_hot',
                                      labeled=True)

                # reset hidden state
                if e == 0:
                    self.h = np.zeros_like(self.h)

                # forward pass
                P, H, loss = self.evaluate(batch,
                                           return_intermediate=True,
                                           return_loss=True)

                # update loss
                if len(loss_smooth) == 0:
                    loss_smooth.append(loss)
                else:
                    loss_smooth.append(loss_smoothing_factor * loss_smooth[-1] + \
                                       (1 - loss_smoothing_factor) * loss)

                # backward pass
                grads = self._gradients(batch, evaluation=(P, H))

                # update parameters
                self.update(grads, eta)

                # update hidden state
                self.h = H[:, -1, np.newaxis]

                # display progress
                if verbose:
                    if (verbose_show_loss and
                        update % verbose_show_loss_frequency == 0):

                        fmt = "iteration {}/{}: loss = {:.3e}"
                        print(fmt.format(update, n_updates, loss_smooth[-1]))

                    if (verbose_show_samples and
                        update % verbose_show_samples_frequency == 0):

                        synth = self.synthesize(
                            x_init=batch.X[:, 0, np.newaxis],
                            length=verbose_show_samples_length)

                        synth = text.get_characters(synth, one_hot=True)

                        synth = '\n'.join(wrap(synth, width=80))

                        fmt = "iteration {}/{}:\n{}\n"
                        print(fmt.format(update, n_updates, synth))

                # check if done
                update += 1
                if update > n_updates:
                    break

        # reset hidden state
        self.h = np.zeros_like(self.h)

        history = TrainHistoryRecurrent(loss_smooth)
        history.add_final_network(self)

        return history

    def _gradients(self, ds, evaluation=None):
        if evaluation is None:
            P, H = self.evaluate(ds, return_intermediate=True)
        else:
            P, H = evaluation

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

        return self._clip_gradients([grad_U, grad_V, grad_W, grad_b, grad_c])

    def _clip_gradients(self, grads, thresh=5):
        return [np.maximum(np.minimum(grad, thresh), -thresh) for grad in grads]
