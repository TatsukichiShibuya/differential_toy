from dttp_layer import *
from net import *
from utils import *

import numpy as np


class dttp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, dataset, epochs, stepsize, backlr):
        # w_plot = np.zeros((epochs, 3))
        for e in range(epochs):
            for x, y in zip(dataset[0], dataset[1]):
                # forward
                y_pred = self.forward(x)
                self.update_backweights(backlr)
                # backward
                self.compute_target(y, y_pred, stepsize)
                self.update_weights(x, y, y_pred)
            # w_plot[e] = self.layers[-1].weight
            # predict
            pred = np.zeros_like(dataset[1])
            for i, x in enumerate(dataset[0]):
                pred[i] = self.predict(x)
            print(f"epoch {e:<4}: {np.sqrt((pred-dataset[1])**2).sum()/2/dataset[0].shape[0]}")

        """
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
        ax.plot(w_plot[:, 0], w_plot[:, 1], w_plot[:, 2], c='b')
        ax.scatter(2.94406835, 1.16284406, 0.66662676, c='k', s=10)
        plt.savefig("weight_3d.png")

        x_plot = np.arange(len(w_plot))
        plt.figure()
        plt.title(f"weight(last layer)")
        plt.plot(x_plot, w_plot, label="F norm")
        plt.legend()
        plt.xlabel("epochs")
        plt.savefig(f"weight.png")
        """

    def update_backweights(self, lr):
        for i in range(self.dim):
            self.layers[i].update_backweight(lr)

    def compute_target(self, y, y_pred, stepsize):
        # initialize
        self.layers[-1].target = y_pred - stepsize * self.loss_derivative(y_pred, y)
        for i in reversed(range(self.dim - 1)):
            self.layers[i].target = self.layers[i + 1].backward(self.layers[i + 1].target)
        # refinement
        repeat = 200

        import matplotlib.pyplot as plt
        x_plot = np.arange(repeat)
        ref_plot = np.zeros((self.dim - 1, repeat))
        tar_plot = np.zeros((self.dim - 1, repeat))
        h_plot = np.zeros((self.dim - 1, repeat))

        for num in range(repeat):
            for i in reversed(range(self.dim - 1)):
                gt = self.layers[i + 1].backward(self.layers[i + 1].target)
                ft_1 = self.layers[i + 1].forward(self.layers[i].target, update=False)
                gft_1 = self.layers[i + 1].backward(ft_1)
                self.layers[i].target += gt - gft_1
        """
                # plot
                ref_plot[i, num] = np.linalg.norm(gt - gft_1)
                tar_plot[i, num] = np.linalg.norm(self.layers[i + 1].target - ft_1)
                h_plot[i, num] = np.linalg.norm(self.layers[i].target - self.layers[i].linear_activation)

        for i in range(self.dim - 1):
            plt.figure()
            plt.title(f"refinement_{i}")
            plt.plot(x_plot, ref_plot[i], label="refinement")
            plt.plot(x_plot, tar_plot[i], label="target")
            plt.legend()
            plt.xlabel("repeat")
            plt.savefig(f"refinement_{i}.png")

        for i in range(self.dim - 1):
            plt.figure()
            plt.title(f"refinement_{i}")
            plt.plot(x_plot, h_plot[i], label="hi-ti")
            plt.legend()
            plt.xlabel("repeat")
            plt.savefig(f"h_{i}.png")
        raise NotImplementedError
        """

    def update_weights(self, x, y, y_pred):
        global_loss = ((self.layers[-1].target - self.layers[-1].linear_activation)**2).sum()
        for i in range(self.dim):
            local_loss = ((self.layers[i].target - self.layers[i].linear_activation)**2).sum()
            lr = global_loss / (local_loss + 1e-9)
            h_previous = self.layers[i - 1].linear_activation if i != 0 else x
            s = self.layers[i].activation_function(h_previous)
            n = s / np.linalg.norm(s)**2
            self.layers[i].weight += lr * (self.layers[i].target -
                                           self.layers[i].linear_activation).reshape(-1, 1)@n.reshape(1, -1)

    def init_layers(self, **kwargs):
        layers = [None] * self.dim
        layers[0] = dttp_layer(in_dim=kwargs["in_dim"],
                               out_dim=kwargs["hid_dim"],
                               activation_function=kwargs["activation_function"],
                               activation_derivative=kwargs["activation_derivative"])
        for i in range(1, self.dim - 1):
            layers[i] = dttp_layer(in_dim=kwargs["hid_dim"],
                                   out_dim=kwargs["hid_dim"],
                                   activation_function=kwargs["activation_function"],
                                   activation_derivative=kwargs["activation_derivative"])
        layers[-1] = dttp_layer(in_dim=kwargs["hid_dim"],
                                out_dim=kwargs["out_dim"],
                                activation_function=(lambda x: x),
                                activation_derivative=(lambda x: 1))
        return layers
