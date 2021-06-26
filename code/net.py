import numpy as np
from utils import *
from layer import *
from abc import ABCMeta, abstractmethod


class net(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.dim = kwargs["dim"]
        self.loss_function = kwargs["loss_function"]
        self.loss_derivative = kwargs["loss_derivative"]
        self.layers = self.init_layers(**kwargs)

    def forward(self, x, update=True):
        y = x
        for i in range(self.dim):
            y = self.layers[i].forward(y, update=update)
        return y

    def predict(self, x):
        return self.forward(x, update=False)

    @abstractmethod
    def init_layers(self):
        raise NotImplementedError
