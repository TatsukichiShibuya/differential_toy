from dttp_layer import *
from net import *
from utils import *

import numpy as np


class dttp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, dataset, epochs, stepsize, backlr):
        for e in range(epochs):
            for x, y in zip(dataset[0], dataset[1]):
                # forward
                y_pred = self.forward(x)
                self.update_backweights(backlr)

                # backward
                self.compute_target(y, y_pred, stepsize)
                self.update_weights(x, y, y_pred)

            # predict
            pred = np.zeros_like(dataset[1])
            for i, x in enumerate(dataset[0]):
                pred[i] = self.predict(x)
            print(f"epoch {e:<4}: {np.sqrt((pred-dataset[1])**2).sum()/2/dataset[0].shape[0]}")

    def update_backweights(self, lr):
        for i in range(self.dim):
            self.layers[i].update_backweight(lr)

    def compute_target(self, y, y_pred, stepsize):
        # initialize
        self.layers[-1].target = self.layers[-1].linear_activation - \
            stepsize * self.loss_derivative(y_pred, y)
        for i in reversed(range(self.dim - 1)):
            self.layers[i].target = self.layers[i + 1].backward(self.layers[i + 1].target)
        # refinement
        pass

    def update_weights(self, x, y, y_pred):
        global_loss = 2 * self.loss_function(y, y_pred)
        for i in range(self.dim):
            local_loss = ((self.layers[i].target - self.layers[i].linear_activation)**2).sum()
            lr = global_loss / local_loss

            h_previous = self.layers[i - 1].linear_activation if i != 0 else x
            s = self.layers[i].activation_function(h_previous)
            n = s / (s**2).sum()

            self.layers[i].weight += lr * (self.layers[i].target -
                                           self.layers[i].linear_activation).reshape(-1, 1)@n.reshape(1, -1)

    def init_layers(self, **kwargs):
        layers = [None] * self.dim
        layers[0] = dttp_layer(in_dim=kwargs["in_dim"],
                               out_dim=kwargs["hid_dim"],
                               activation_function=kwargs["activation_function"],
                               activation_derivative=kwargs["activation_derivative"])
        for i in range(1, self.dim - 1):
            layers[i] = dttp_layer(in_dim=kwargs["hid_dim"],
                                   out_dim=kwargs["hid_dim"],
                                   activation_function=kwargs["activation_function"],
                                   activation_derivative=kwargs["activation_derivative"])
        layers[-1] = dttp_layer(in_dim=kwargs["hid_dim"],
                                out_dim=kwargs["out_dim"],
                                activation_function=(lambda x: x),
                                activation_derivative=(lambda x: 1))
        return layers
