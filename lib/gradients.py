import numpy as np


def gradient_error(grad, grad_num):
    eps = np.full_like(grad, np.spacing(1))

    d_abs = np.abs(grad - grad_num)
    d_rel = d_abs / np.maximum(eps, np.abs(grad) + np.abs(grad_num))

    fmt = "{} difference: max = {:1.2e}, mean = {:1.2e}, std = {:1.2e}"
    for d, title in (d_abs, "Absolute"), (d_rel, "Relative"):
        print(fmt.format(title, d.max(), d.mean(), d.std()))


def compare_gradients(network_constructor, ds, params):
    for n, dims, alpha in params:
        network = network_constructor(dims, ds.num_classes, alpha=alpha)
        ds_sub = ds.subsample(n=n, dims=dims)

        grad_W_num, grad_b_num = network.gradients(ds_sub, numerical=True)
        grad_W, grad_b = network.gradients(ds_sub)

        print(f"{dims} dimensions, {n} sample(s), lambda = {alpha}:\n")

        print("W:")
        gradient_error(grad_W, grad_W_num)

        print()

        print("b:")
        gradient_error(grad_b, grad_b_num)

        print()
