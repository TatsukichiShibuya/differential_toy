from layer import *
import numpy as np


class dttp_layer(layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target = None

    def forward(self, x, update=True):
        a = self.activation_function(x)
        h = self.weight@a
        if update:
            self.activation = a
            self.linear_activation = h
            n = self.activation_function(h) / np.linalg.norm(self.activation_function(h))**2
            self.backweight_grad = (x - self.backward(h)).reshape(-1, 1)@n.reshape(1, -1)
        return h

    def backward(self, x):
        a = self.activation_function(x)
        h = self.backweight@a
        return h

    def update_backweight(self, lr):
        self.backweight += lr * self.backweight_grad
