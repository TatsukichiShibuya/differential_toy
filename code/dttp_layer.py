import numpy as np


class dttp_layer:
    def __init__(self, **kwargs):
        self.weight = np.random.randn(kwargs["out_dim"], kwargs["in_dim"])
        self.backweight = np.random.randn(kwargs["in_dim"], kwargs["out_dim"])
        self.backweight_grad = None
        self.activation_function = kwargs["activation_function"]
        self.activation_derivative = kwargs["activation_derivative"]

        self.target = None

        self.activation = None
        self.linear_activation = None

    def forward(self, x, update=True):
        a = self.activation_function(x)
        h = self.weight@a
        if update:
            self.activation = a
            self.linear_activation = h
            n = self.activation_function(h) / (self.activation_function(h)**2).sum()
            self.backweight_grad = (x - self.backward(h)).reshape(-1, 1)@n.reshape(1, -1)
        return h

    def backward(self, x):
        a = self.activation_function(x)
        h = self.backweight@a
        return h

    def update_backweight(self, lr):
        self.backweight += lr * self.backweight_grad
