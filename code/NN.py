import numpy as np
from utils import *
from abc import ABCMeta, abstractmethod


class NN(metaclass=ABCMeta):
    def __init__(self, dim, in_dim, out_dim, hid_dim):
        self.dim = dim
        self.act = (lambda x: leakyrelu(x, a=0.2))
        self.act_dash = (lambda x: leakyrelu_dash(x, a=0.2))
        self.loss = (lambda y, y_: ((y - y_)**2).sum() / 2)

        self.weights = [None] * dim
        self.weights[0] = np.random.randn(hid_dim, in_dim)
        for i in range(1, dim - 1):
            self.weights[i] = np.random.randn(hid_dim, hid_dim)
        self.weights[dim - 1] = np.random.randn(out_dim, hid_dim)

    @abstractmethod
    def train(self, dataset, epochs, lr):
        raise NotImplementedError()

    def test(self, x):
        y = x
        for i in range(self.dim):
            y = self.act(self.weights[i]@y)
        return y
