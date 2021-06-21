import numpy as np


class layer:
    def __init__(self, **kwargs):
        self.weight = np.random.randn(kwargs["in_dim"], kwargs["out_dim"])
        self.activation_func = None

    def forward(self):
        pass
