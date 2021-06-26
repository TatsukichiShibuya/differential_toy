import numpy as np
import torch
from torch import nn
from abc import ABCMeta, abstractmethod


class layer(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        # weights
        torch.manual_seed(4)
        w = torch.empty(kwargs["out_dim"], kwargs["in_dim"])
        nn.init.orthogonal_(w)
        self.weight = w.to('cpu').detach().numpy().copy()
        self.weight_grad = None
        self.backweight = w.T.to('cpu').detach().numpy().copy()
        self.backweight_grad = None

        # functions
        self.activation_function = kwargs["activation_function"]
        self.activation_derivative = kwargs["activation_derivative"]

        # activation
        self.activation = None
        self.linear_activation = None

    @abstractmethod
    def forward(self, x, update=True):
        raise NotImplementedError
