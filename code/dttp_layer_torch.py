import torch
from torch import nn


class dttp_layer_torch:
    def __init__(self, **kwargs):
        # weights
        torch.manual_seed(4)
        self.weight = torch.empty(kwargs["out_dim"], kwargs["in_dim"], requires_grad=True)
        nn.init.orthogonal_(self.weight)
        self.backweight = torch.empty(kwargs["out_dim"], kwargs["in_dim"], requires_grad=True)
        nn.init.orthogonal_(self.backweight)

        # grad
        self.weight_grad = None
        self.backweight_grad = None

        # functions
        self.activation_function = kwargs["activation_function"]
        self.activation_derivative = kwargs["activation_derivative"]

        # activation
        self.activation = torch.empty(kwargs["in_dim"], 1, requires_grad=True)
        self.linear_activation = torch.empty(kwargs["out_dim"], 1, requires_grad=True)

        # target
        self.target = None

        # regular
        self.regular = (kwargs["in_dim"] == kwargs["out_dim"])

    def forward(self, x, update=True):
        if update:
            self.activation = self.activation_function(x)
            self.linear_activation = self.weight@self.activation
            self.linear_activation.retain_grad()
            if self.regular:
                s = self.activation_function(self.linear_activation)
                n = s / torch.norm(s)**2
                self.backweight_grad = (x - self.backward(self.linear_activation)
                                        ).reshape(-1, 1)@n.reshape(1, -1)
            return self.linear_activation
        else:
            a = self.activation_function(x)
            h = self.weight@a
            return h

    def backward(self, x):
        a = self.activation_function(x)
        h = self.backweight@a
        return h

    def update_backweight(self, lr):
        if self.backweight_grad is not None:
            with torch.no_grad():
                self.backweight += lr * self.backweight_grad
