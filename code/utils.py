import numpy as np


def leakyrelu(x, a):
    x_copy = x.copy()
    frag = x < 0
    x_copy[frag] = a * x_copy[frag]
    return x_copy


def leakyrelu_dash(x, a):
    x_copy = x.copy()
    frag = x < 0
    x_copy[frag] = a
    x_copy[~frag] = 1
    return x_copy


def make_dataset(sigma=1, seed=1):
    # f(x) = sin(x)+0.5x
    np.random.seed(seed)
    x = np.arange(0, 10.1, 0.1)
    y = np.sin(x) + x / 2 + np.random.normal(0, sigma, x.shape)
    return [x, y]
