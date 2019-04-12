import numpy as np

from network import NUM_GRAD_DELTA


def gradient_error(grad, grad_num):
    eps = np.full_like(grad, np.spacing(1))

    d_abs = np.abs(grad - grad_num)
    d_rel = d_abs / np.maximum(eps, np.abs(grad) + np.abs(grad_num))

    fmt = "{} difference: max = {:1.2e}, mean = {:1.2e}, std = {:1.2e}"
    for d, title in (d_abs, "Absolute"), (d_rel, "Relative"):
        print(fmt.format(title, d.max(), d.mean(), d.std()))


def compare_gradients(network_constructor,
                      ds,
                      params,
                      h=NUM_GRAD_DELTA,
                      random_seed=None):

    for n, dims, alpha in params:
        network = network_constructor(input_size=dims,
                                      num_classes=ds.num_classes,
                                      alpha=alpha,
                                      random_seed=random_seed)

        ds_sub = ds.subsample(n=n, dims=dims)

        grads_num = network.gradients(ds_sub, numerical=True, h=h)
        grads = network.gradients(ds_sub)

        fmt = "{} dimensions, {} sample(s), lambda = {}:\n"
        msg = fmt.format(dims, n, alpha)
        print(msg)

        for grad_num, grad, param in zip(grads_num, grads, network.param_names):
            print(param + ':')
            gradient_error(grad, grad_num)
            print()


def compare_gradients_recurrent(network_constructor,
                                ds,
                                h=NUM_GRAD_DELTA,
                                random_seed=None):

    network = network_constructor(random_seed=random_seed)

    grads_num = network.gradients(ds, numerical=True, h=h)
    grads = network.gradients(ds)

    for grad_num, grad, param in zip(grads_num, grads, network.param_names):
        print(param + ':')
        gradient_error(grad, grad_num)
        print()
