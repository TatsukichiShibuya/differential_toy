from layer import *
import numpy as np


class bp_layer(layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x, update=True):
        a = self.weight@x
        h = self.activation_function(a)
        if update:
            self.linear_activation = a
            self.activation = h
        return h

    def backward(self, g, h_previous, update=True):
        ret = g * self.activation_derivative(self.linear_activation)
        if update:
            self.weight_grad = ret.reshape(-1, 1)@h_previous.reshape(1, -1)
        ret = self.weight.T@ret
        return ret

    def update_weight(self, lr):
        self.weight -= lr * self.weight_grad
