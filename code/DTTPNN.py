from NN import *
from utils import *
import numpy as np


class DTTPNN(NN):
    def __init__(self, dim, in_dim, out_dim, hid_dim):
        super().__init__(dim, in_dim, out_dim, hid_dim)

    def train(self, dataset, epochs, lr):
        pass
