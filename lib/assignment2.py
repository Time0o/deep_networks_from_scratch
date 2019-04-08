import matplotlib.pyplot as plt
import numpy as np


def visualize_learning_curves(history):
    us = np.arange(1, len(history.learning_rate) + 1)
    us_ss = np.linspace(1, len(history.learning_rate) + 1, history.length)

    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    (ax_eta, ax_cost), (ax_loss, ax_acc) = axes

    ax_eta.plot(us, history.learning_rate)
    ax_eta.set_title("Learning Rate Schedule")
    ax_eta.set_ylabel(r"$\eta$")

    ax_cost.plot(us_ss, history.train_cost, label="Training")
    ax_cost.plot(us_ss, history.val_cost, label="Validation")
    ax_cost.set_title("Cost")
    ax_cost.set_ylabel("Cost")
    ax_cost.legend()

    ax_loss.plot(us_ss, history.train_loss, label="Training")
    ax_loss.plot(us_ss, history.val_loss, label="Validation")
    ax_loss.set_title("Loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    ax_acc.plot(us_ss, history.train_accuracy, label="Training")
    ax_acc.plot(us_ss, history.val_accuracy, label="Validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    for ax in axes.flatten():
        ax.set_xlabel("Update Step")
        ax.set_xlim([1, len(history.learning_rate)])
        ax.grid()

    plt.tight_layout()
