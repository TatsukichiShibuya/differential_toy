from net import *
from utils import *
import numpy as np


class dttp_net(NN):
    def __init__(self, dim, in_dim, out_dim, hid_dim):
        super().__init__(dim, in_dim, out_dim, hid_dim)

        self.back_act = [None] * self.dim
        for i in range(self.dim - 1):
            self.back_act[i] = (lambda x: leakyrelu(x, a=0.2))
        self.back_act[-1] = (lambda x: x)

        self.back_weights = [None] * self.dim
        self.back_weights[0] = np.random.randn(hid_dim, in_dim)
        for i in range(1, dim - 1):
            self.back_weights[i] = np.random.randn(hid_dim, hid_dim)
        self.back_weights[dim - 1] = np.random.randn(out_dim, hid_dim)

        self.target = [None] * self.dim

    def train(self, dataset, epochs, lr):
        for e in range(epochs):
            for x, y in zip(dataset[0], dataset[1]):
                forward(x)
                calc_target()
                backward()

    def forward(self, x, lr, update_backweights=True):
        # forward input and update back_weights
        a = [None] * (self.dim + 1)
        h = [None] * (self.dim + 1)
        h[0] = x
        for i in range(self.dim):
            a[i + 1] = self.act[i](h[i])
            h[i + 1] = self.weights[i]@a[i + 1]

    def calc_target(self):
        # calc and refine target
        pass

    def backward(self, update_weights=True):
        # update weights using target
        pass
