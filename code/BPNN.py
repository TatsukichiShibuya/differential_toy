from NN import *
from utils import *
import numpy as np


class BPNN(NN):
    def __init__(self, dim, in_dim, out_dim, hid_dim):
        super().__init__(dim, in_dim, out_dim, hid_dim)

    def train(self, dataset, epochs, lr):
        for e in range(epochs):
            for x, y in zip(dataset[0], dataset[1]):
                # forward
                h = [None] * (self.dim + 1)
                a = [None] * (self.dim + 1)
                h[0] = x
                for i in range(self.dim):
                    a[i + 1] = self.weights[i]@h[i]
                    h[i + 1] = self.act(a[i + 1])
                loss = self.loss(h[-1], y)
                # backward
                grads = [None] * self.dim
                g = (h[-1] - y)
                for i in reversed(range(self.dim)):
                    g = g * self.act_dash(a[i + 1])
                    grads[i] = g.reshape(-1, 1)@h[i].reshape(1, -1)
                    g = self.weights[i].T@g
                # update
                for i in range(self.dim):
                    self.weights[i] -= lr * grads[i]
            print(f"epochs {e}: {loss}")
