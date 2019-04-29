import matplotlib.pyplot as plt
import numpy as np

from history import TrainHistory


def _history_postfix(sigma, bn):
    fmt = 'three_layers_sigma_e{}_{}bn'
    return fmt.format(int(np.log(sigma)), '' if bn else 'no_')


def train_batchnorm_stabilization(train_function, sigmas, pickle_dir):
    for sigma in sigmas:
        for bn in False, True:
            history = train_function(sigma, bn)

            history.save(pickle_dir, postfix=_history_postfix(sigma, bn))


def visualize_batchnorm_stabilization(sigmas, pickle_dir):
    _, axes = plt.subplots(len(sigmas), 2, figsize=(10, 5 * len(sigmas)))

    for i, sigma in enumerate(sigmas):
        for j, bn in enumerate([False, True]):
            history = TrainHistory.load(pickle_dir,
                                        postfix=_history_postfix(sigma, bn))

            us = np.linspace(1, len(history.learning_rate) + 1, history.length)

            axes[i, j].plot(us, history.train_loss)

            axes[i, j].set_title(r"$\sigma = {:.1e}$, {}".format(
                sigma, 'bn.' if bn else 'no bn.'))

            axes[i, j].grid()
