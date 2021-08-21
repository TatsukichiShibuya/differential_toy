from dctp_layer import *
from net import *
from utils import *

import numpy as np


class dctp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, dataset, epochs, stepsize, lrb):
        for e in range(epochs):
            for x, y in zip(dataset[0], dataset[1]):
                # train backward
                y_pred = self.forward(x)
                for i in range(10):
                    for d in range(1, self.dim - 1):
                        h = self.layers[d].activation.detach()
                        h += torch.normal(0, 1, size=h.shape)
                        loss = torch.norm(self.layers[d].backward(
                            self.layers[d].forward(h, update=False)) - h)**2
                        self.zero_grad()
                        loss.backward()
                        with torch.no_grad():
                            self.layers[d].backweight -= lrb * self.layers[d].backweight.grad

                # compute target
                y_pred = self.forward(x)
                self.compute_target(y, y_pred, stepsize)

                # train forward
                self.update_weight(x)

            # predict
            pred = torch.zeros_like(dataset[1])
            for i, x in enumerate(dataset[0]):
                pred[i] = self.predict(x)
            print(f"epoch {e:<4}: {(torch.norm(pred-dataset[1])**2)/(2*len(dataset[0]))}")

    def compute_target(self, y, y_pred, stepsize):
        loss = torch.norm(y_pred - y)**2
        self.zero_grad()
        loss.backward()
        with torch.no_grad():
            for d in range(self.dim - 2, self.dim):
                self.layers[d].target = self.layers[d].activation - \
                    stepsize * self.layers[d].activation.grad
            for d in reversed(range(self.dim - 2)):
                self.layers[d].target = self.layers[d + 1].backward(self.layers[d + 1].target)
                self.layers[d].target += self.layers[d].activation
                self.layers[d].target -= self.layers[d + 1].backward(self.layers[d + 1].activation)

    def update_weight(self, x):
        for d in range(self.dim):
            h_previous = self.layers[d - 1].activation.detach() if d != 0 else x
            h = self.layers[d].forward(h_previous, update=False)
            t = self.layers[d].target.detach()
            loss = torch.norm(h - t)**2
            self.zero_grad()
            loss.backward()
            with torch.no_grad():
                self.layers[d].weight -= self.layers[d].weight.grad

    def zero_grad(self):
        for d in range(self.dim):
            if self.layers[d].activation.grad is not None:
                self.layers[d].activation.grad.zero_()
            if self.layers[d].weight.grad is not None:
                self.layers[d].weight.grad.zero_()
            if self.layers[d].backweight.grad is not None:
                self.layers[d].backweight.grad.zero_()

    def init_layers(self, **kwargs):
        layers = [None] * self.dim
        layers[0] = dctp_layer(in_dim=kwargs["in_dim"],
                               out_dim=kwargs["hid_dim"],
                               activation_function=kwargs["activation_function"],
                               activation_derivative=kwargs["activation_derivative"])
        for i in range(1, self.dim - 1):
            layers[i] = dctp_layer(in_dim=kwargs["hid_dim"],
                                   out_dim=kwargs["hid_dim"],
                                   activation_function=kwargs["activation_function"],
                                   activation_derivative=kwargs["activation_derivative"])
        layers[-1] = dctp_layer(in_dim=kwargs["hid_dim"],
                                out_dim=kwargs["out_dim"],
                                activation_function=(lambda x: x),
                                activation_derivative=(lambda x: 1))
        return layers
