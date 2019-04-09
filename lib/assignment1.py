from history import TrainHistory
from single_layer_fully_connected import SingleLayerFullyConnected

import matplotlib.pyplot as plt


def _axes(rows, cols, figsize=None):
    if figsize is None:
        figsize = (5 * cols, 5 * rows)

    _, axes = plt.subplots(rows, cols, figsize=figsize)

    return axes


def train_networks(ds_train,
                   ds_val,
                   hyperparams,
                   pickle_dir,
                   postfix=None,
                   random_seed=None):

    loss = hyperparams.get('loss', 'cross_entropy')

    for alpha, eta in zip(hyperparams['alpha'], hyperparams['eta']):
        network = SingleLayerFullyConnected(ds_train.input_size,
                                            ds_train.num_classes,
                                            alpha=alpha,
                                            loss=loss,
                                            random_seed=random_seed)

        history = network.train(ds_train,
                                ds_val,
                                eta=eta,
                                n_epochs=hyperparams['epochs'],
                                verbose=True)

        history.save(pickle_dir,
                     [('alpha', alpha), ('eta', eta)],
                     postfix=postfix)


def load_histories(hyperparams, pickle_dir, postfix=None):
    histories = []

    for alpha, eta in zip(hyperparams['alpha'], hyperparams['eta']):
        history = TrainHistory.load(pickle_dir,
                                    [('alpha', alpha), ('eta', eta)],
                                    postfix=postfix)

        history.add_title(r"$\lambda = {}$, $\eta = {}$".format(alpha, eta))

        histories.append(history)

    return histories


def visualize_learning_curves(histories):
    axes = _axes(len(histories), 2)

    for i, history in enumerate(histories):
        history.visualize(axes=axes[i, :])


def visualize_performance(histories, ds):
    axes = _axes(len(histories) // 2, 2)

    for history, ax in zip(histories, axes.flatten()):
        history.final_network.visualize_performance(
           ds, ax=ax, title=history.title)


def visualize_weights(histories, ds):
    axes = _axes(len(histories),
                 ds.num_classes,
                 figsize=(ds.num_classes, len(histories)))

    for i, history in enumerate(histories):
        history.final_network.visualize_weights(axes=axes[i, :])

        axes[i, 0].set_ylabel(history.title, labelpad=50, rotation=0)

    for ax, label in zip(axes[-1, :], ds.labels):
        ax.set_xlabel(label, rotation=45)
