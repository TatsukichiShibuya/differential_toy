import numpy as np
from utils import *
from abc import ABCMeta, abstractmethod
from layer import *
"""
TODO:
・初期化をorthogonalに変える
・活性化関数とかは引数にする
・getargsを実装する

"""


class net(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.dim = kwargs["dim"]
        activation_funcion = kwargs["activation_funcion"]
        activation_derivative = kwargs["activation_derivative"]

        self.layers = [None] * self.dim
        self.layers[i] = layer(in_dim, hid_dim)
        for i in range(1, self.dim - 1):
            self.layers[i] = layer(hid_dim, hid_dim)
        self.layers[self.dim - 1] = layer(hid_dim, out_dim)

        self.act = [None] * self.dim
        for i in range(self.dim - 1):
            self.act[i] = (lambda x: leakyrelu(x, a=0.2))
        self.act[-1] = (lambda x: x)

        self.act_dash = [None] * self.dim
        for i in range(self.dim - 1):
            self.act_dash[i] = 1
        self.act_dash[-1] = (lambda x: 1)

        self.loss = (lambda y, y_: ((y - y_)**2).sum() / 2)

    @abstractmethod
    def train(self, dataset, epochs, lr):
        raise NotImplementedError()

    def predict(self, x):
        y = x
        for i in range(self.dim):
            y = self.act(self.weights[i]@y)
        return y
