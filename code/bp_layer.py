import numpy as np


class bp_layer:
    def __init__(self, **kwargs):
        self.weight = np.random.randn(kwargs["out_dim"], kwargs["in_dim"])
        self.weight_grad = None
        self.activation_function = kwargs["activation_function"]
        self.activation_derivative = kwargs["activation_derivative"]

        self.linear_activation = None
        self.activation = None

    def forward(self, x, update=True):
        a = self.weight@x
        h = self.activation_function(a)
        if update:
            self.linear_activation = a
            self.activation = h
        return h

    def backward(self, g, h_previous, update=True):
        # g: nabla_{h_current}L
        # ret: nabla_{W_current}L
        ret = g * self.activation_derivative(self.linear_activation)
        if update:
            self.weight_grad = ret.reshape(-1, 1)@h_previous.reshape(1, -1)
        ret = self.weight.T@ret
        return ret

    def update_weight(self, lr):
        self.weight -= lr * self.weight_grad
