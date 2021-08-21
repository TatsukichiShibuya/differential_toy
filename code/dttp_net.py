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
                # self.compute_target(y, y_pred, stepsize)
                self.compute_target_regular(y, y_pred, stepsize)
                self.update_weights(x, y, y_pred)
            # predict
            pred = np.zeros_like(dataset[1])
            for i, x in enumerate(dataset[0]):
                pred[i] = self.predict(x)
            print(f"epoch {e:<4}: {(np.linalg.norm(pred-dataset[1])**2)/(2*len(dataset[0]))}")

    def update_backweights(self, lr):
        for i in range(self.dim):
            self.layers[i].update_backweight(lr)

    def compute_target(self, y, y_pred, stepsize):
        # initialize
        self.layers[-1].target = y_pred - stepsize * self.loss_derivative(y_pred, y)
        for i in reversed(range(self.dim - 1)):
            self.layers[i].target = self.layers[i + 1].backward(self.layers[i + 1].target)

        # refinement
        repeat = 200

        for num in range(repeat):
            for i in reversed(range(self.dim - 1)):
                gt = self.layers[i + 1].backward(self.layers[i + 1].target)
                ft_1 = self.layers[i + 1].forward(self.layers[i].target, update=False)
                gft_1 = self.layers[i + 1].backward(ft_1)
                self.layers[i].target += gt - gft_1

    def compute_target_regular(self, y, y_pred, stepsize):
        y_pred2 = self.layers[-2].linear_activation
        # initialize
        dLdhL = self.loss_derivative(y_pred, y)
        self.layers[-1].target = y_pred - stepsize * dLdhL
        dLdhL1 = np.diag(self.layers[-1].activation_derivative(y_pred2)
                         )@self.layers[-1].weight.T@self.loss_derivative(y_pred, y)
        self.layers[-2].target = y_pred2 - stepsize * dLdhL1
        for i in reversed(range(self.dim - 2)):
            self.layers[i].target = self.layers[i + 1].backward(self.layers[i + 1].target)

        # refinement
        repeat = 200
        for num in range(repeat):
            for i in reversed(range(self.dim - 2)):
                gt = self.layers[i + 1].backward(self.layers[i + 1].target)
                ft_1 = self.layers[i + 1].forward(self.layers[i].target, update=False)
                gft_1 = self.layers[i + 1].backward(ft_1)
                self.layers[i].target += gt - gft_1

    def update_weights(self, x, y, y_pred):
        global_loss = ((self.layers[-1].target - self.layers[-1].linear_activation)**2).sum()
        for i in range(self.dim):
            local_loss = ((self.layers[i].target - self.layers[i].linear_activation)**2).sum()
            lr = global_loss / (local_loss + 1e-12)
            h_previous = self.layers[i - 1].linear_activation if i != 0 else x
            s = self.layers[i].activation_function(h_previous)
            n = s / np.linalg.norm(s)**2
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
                                activation_function=kwargs["activation_function"],
                                activation_derivative=kwargs["activation_derivative"])
        return layers
