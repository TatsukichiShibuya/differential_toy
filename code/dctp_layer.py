import numpy as np
import torch
from torch import nn


class dctp_layer:
    def __init__(self, **kwargs):
        # weights
        torch.manual_seed(4)
        self.weight = torch.empty(kwargs["out_dim"], kwargs["in_dim"], requires_grad=True)
        nn.init.orthogonal_(self.weight)
        self.backweight = torch.empty(kwargs["out_dim"], kwargs["in_dim"], requires_grad=True)
        nn.init.orthogonal_(self.backweight)

        # functions
        self.activation_function = kwargs["activation_function"]
        self.activation_derivative = kwargs["activation_derivative"]

        # activation
        self.activation = torch.empty(kwargs["out_dim"], 1, requires_grad=True)
        self.linear_activation = torch.empty(kwargs["out_dim"], 1, requires_grad=True)

        # target
        self.target = None

    def forward(self, x, update=True):
        if update:
            self.linear_activation = self.weight@x
            self.activation = self.activation_function(self.linear_activation)
            self.activation.retain_grad()
            return self.activation
        else:
            a = self.weight@x
            h = self.activation_function(a)
            return h

    def backward(self, x):
        a = self.backweight@x
        h = self.activation_function(a)
        return h
