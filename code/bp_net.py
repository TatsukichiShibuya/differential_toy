from utils import *
import numpy as np
from bp_layer import *
"""
TODO:
・backwardをカプセル
"""


class bp_net:
    def __init__(self, **kwargs):
        self.dim = kwargs["dim"]
        activation_function = kwargs["activation_function"]
        activation_derivative = kwargs["activation_derivative"]

        self.layers = [None] * self.dim
        self.layers[0] = bp_layer(in_dim=kwargs["in_dim"],
                                  out_dim=kwargs["hid_dim"],
                                  activation_function=activation_function,
                                  activation_derivative=activation_derivative)
        for i in range(1, self.dim - 1):
            self.layers[i] = bp_layer(in_dim=kwargs["hid_dim"],
                                      out_dim=kwargs["hid_dim"],
                                      activation_function=activation_function,
                                      activation_derivative=activation_derivative)
        self.layers[self.dim - 1] = bp_layer(in_dim=kwargs["hid_dim"],
                                             out_dim=kwargs["out_dim"],
                                             activation_function=(lambda x: x),
                                             activation_derivative=(lambda x: 1))

        self.loss_function = kwargs["loss_function"]
        self.loss_derivative = kwargs["loss_derivative"]

    def train(self, dataset, epochs, lr):
        for e in range(epochs):
            for x, y in zip(dataset[0], dataset[1]):
                # forward
                y_pred = self.forward(x)
                # backward
                grads = [None] * self.dim
                g = self.loss_derivative(y_pred, y)
                for i in reversed(range(1, self.dim)):
                    g = g * self.layers[i].activation_derivative(self.layers[i].linear_activation)
                    self.layers[i].weight_grad = g.reshape(-1,
                                                           1)@self.layers[i - 1].activation.reshape(1, -1)
                    g = self.layers[i].weight.T@g
                self.layers[0].weight_grad = g.reshape(-1, 1)@x.reshape(1, -1)
                # update weights
                for i in range(self.dim):
                    self.layers[i].update_weight(lr)

            pred = np.zeros_like(dataset[1])
            for i, x in enumerate(dataset[0]):
                pred[i] = self.predict(x)
            print(f"epoch {e}: {np.sqrt((pred-dataset[1])**2).sum()/2/dataset[0].shape[0]}")

    def forward(self, x, update=True):
        y = x
        for i in range(self.dim):
            y = self.layers[i].forward(y, update=update)
        return y

    def predict(self, x):
        return self.forward(x, update=False)
