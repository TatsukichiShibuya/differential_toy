from utils import *
from bp_layer import *
from net import *

import numpy as np


class bp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, dataset, epochs, lr):
        for e in range(epochs):
            for x, y in zip(dataset[0], dataset[1]):
                # forward
                y_pred = self.forward(x)

                # backward
                self.backward(x, y, y_pred)
                self.update_weights(lr)

            # predict
            pred = np.zeros_like(dataset[1])
            for i, x in enumerate(dataset[0]):
                pred[i] = self.predict(x)
            print(f"epoch {e:<4}: {(np.linalg.norm(pred-dataset[1])**2)/(2*len(dataset[0]))}")

    def backward(self, x, y, y_pred, update=True):
        grads = [None] * self.dim
        g = self.loss_derivative(y_pred, y)
        for i in reversed(range(1, self.dim)):
            g = self.layers[i].backward(g, self.layers[i - 1].activation, update=update)
        self.layers[0].backward(g, x, update=update)

    def update_weights(self, lr):
        for i in range(self.dim):
            self.layers[i].update_weight(lr)

    def init_layers(self, **kwargs):
        layers = [None] * self.dim
        layers[0] = bp_layer(in_dim=kwargs["in_dim"],
                             out_dim=kwargs["hid_dim"],
                             activation_function=kwargs["activation_function"],
                             activation_derivative=kwargs["activation_derivative"])
        for i in range(1, self.dim - 1):
            layers[i] = bp_layer(in_dim=kwargs["hid_dim"],
                                 out_dim=kwargs["hid_dim"],
                                 activation_function=kwargs["activation_function"],
                                 activation_derivative=kwargs["activation_derivative"])
        layers[-1] = bp_layer(in_dim=kwargs["hid_dim"],
                              out_dim=kwargs["out_dim"],
                              activation_function=(lambda x: x),
                              activation_derivative=(lambda x: 1))
        return layers
