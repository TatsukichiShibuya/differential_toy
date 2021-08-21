from dttp_layer import *
from net import *
from utils import *

import numpy as np
import matplotlib.pyplot as plt


class dttp_net(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w_norm_plot = np.zeros((0))
        self.lr_plot = np.zeros((0))

    def train(self, dataset, epochs, stepsize, backlr):
        # w_plot = np.zeros((epochs, 3))
        for e in range(epochs):
            for x, y in zip(dataset[0], dataset[1]):

                for e_ in range(50):
                    x_ = np.random.sample(x.shape) * 10
                    y_pred = self.forward(x_)
                    self.update_backweights(backlr)

                # forward
                y_pred = self.forward(x)
                self.update_backweights(backlr)
                # backward
                #self.compute_target(y, y_pred, stepsize)
                self.compute_target_regular(y, y_pred, stepsize)
                self.update_weights(x, y, y_pred)
                # self.update_weights_regular(x, y, y_pred)
            # w_plot[e] = self.layers[-1].weight
            # predict
            pred = np.zeros_like(dataset[1])
            for i, x in enumerate(dataset[0]):
                pred[i] = self.predict(x)

            print(f"epoch {e:<4}: {(np.linalg.norm(pred-dataset[1])**2)/(2*len(dataset[0]))}")
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
        print("grad norm")
        for i in range(self.dim):
            print(i, self.w_norm_plot[i::self.dim].mean(), self.w_norm_plot[i::self.dim].std())
        print("lr")
        for i in range(self.dim - 1):
            print(i, self.lr_plot[i::self.dim - 1].mean(), self.lr_plot[i::self.dim - 1].std())

    def train_full(self, dataset, epochs, stepsize, backlr):
        # w_plot = np.zeros((epochs, 3))
        for e in range(epochs):
            grad_mean = [None] * self.dim
            for x, y in zip(dataset[0], dataset[1]):
                # train backward
                # print("before:", self.reconstruction_loss(x))
                for e_ in range(50):
                    x_ = np.random.sample(x.shape) * 10
                    y_pred = self.forward(x_)
                    self.update_backweights(backlr)
                # print("after:", self.reconstruction_loss(x))

                # forward
                y_pred = self.forward(x)
                self.update_backweights(backlr)

                # backward
                # self.compute_target(y, y_pred, stepsize)
                suc = self.compute_target_regular(y, y_pred, stepsize)
                """if not suc:
                    print(x, y)"""
                # self.update_weights(x, y, y_pred)
                grad = self.compute_grad(x, y, y_pred)
                for i in range(self.dim):
                    if grad_mean[i] is None:
                        grad_mean[i] = grad[i]
                    else:
                        grad_mean[i] += grad[i]
                # self.update_weights_regular(x, y, y_pred)
            # w_plot[e] = self.layers[-1].weight
            for i in range(self.dim - 1, self.dim):
                self.layers[i].weight += grad_mean[i] / len(dataset[0])

            # predict
            pred = np.zeros_like(dataset[1])
            for i, x in enumerate(dataset[0]):
                pred[i] = self.predict(x)
            print(f"epoch {e:<4}: {(np.linalg.norm(pred-dataset[1])**2)/(2*len(dataset[0]))}")
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
        print("grad norm")
        for i in range(self.dim):
            print(i, self.w_norm_plot[i::self.dim].mean(), self.w_norm_plot[i::self.dim].std())
        print("lr")
        for i in range(self.dim - 1):
            print(i, self.lr_plot[i::self.dim - 1].mean(), self.lr_plot[i::self.dim - 1].std())

    def train_last(self, dataset, epochs, stepsize, backlr):
        for e in range(epochs):
            dLdh = None
            for x, y in zip(dataset[0], dataset[1]):
                # train backward
                # print("before:", self.reconstruction_loss(x))
                for e_ in range(0):
                    x_ = np.random.sample(x.shape) * 10
                    y_pred = self.forward(x_)
                    self.update_backweights(backlr)
                # print("after:", self.reconstruction_loss(x))

                # forward
                y_pred = self.forward(x)
                # self.update_backweights(backlr)
                if dLdh is None:
                    dLdh = self.loss_derivative(y_pred, y)
                else:
                    dLdh += self.loss_derivative(y_pred, y)

            # backward
            print(dLdh)
            target = y_pred - stepsize * dLdh / len(dataset[0])
            n = self.layers[-1].activation / np.linalg.norm(self.layers[-1].activation)**2
            grad = (target - y_pred).reshape(-1, 1)@n.reshape(1, -1)

            # update
            self.layers[-1].weight += grad

            # predict
            pred = np.zeros_like(dataset[1])
            for i, x in enumerate(dataset[0]):
                pred[i] = self.predict(x)
            print(f"epoch {e:<4}: {(np.linalg.norm(pred-dataset[1])**2)/(2*len(dataset[0]))}")

    def train_last_full(self, dataset, epochs, stepsize, backlr):
        size = len(dataset[0])
        for e in range(epochs):
            loss = 0
            dLdh = 0
            for x, y in zip(dataset[0], dataset[1]):
                # forward
                y_pred = self.forward(x)
                if dLdh is None:
                    dLdh = self.loss_derivative(y_pred, y)
                else:
                    dLdh += self.loss_derivative(y_pred, y)
            correction = dLdh * stepsize / size * (size - 1) / size
            grad = None
            for x, y in zip(dataset[0], dataset[1]):
                # forward
                y_pred = self.forward(x)
                target = y_pred - stepsize * self.loss_derivative(y_pred, y)  # + correction
                n = self.layers[-1].activation / np.linalg.norm(self.layers[-1].activation)**2
                if grad is None:
                    grad = (target - y_pred).reshape(-1, 1)@n.reshape(1, -1)
                else:
                    grad += (target - y_pred).reshape(-1, 1)@n.reshape(1, -1)
            # update
            self.layers[-1].weight += grad / size
            # predict
            pred = np.zeros_like(dataset[1])
            for i, x in enumerate(dataset[0]):
                pred[i] = self.predict(x)
            print(f"epoch {e:<4}: {(np.linalg.norm(pred-dataset[1])**2)/(2*len(dataset[0]))}")

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
                h_plot[i, num] = np.linalg.norm(
                    self.layers[i].target - self.layers[i].linear_activation)

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

    def compute_target_regular(self, y, y_pred, stepsize):
        y_pred2 = self.layers[-2].linear_activation
        # initialize
        dLdhL = self.loss_derivative(y_pred, y)
        self.layers[-1].target = y_pred - stepsize * dLdhL
        dLdhL1 = np.diag(self.layers[-1].activation_derivative(y_pred2)
                         )@self.layers[-1].weight.T@self.loss_derivative(y_pred, y)
        self.layers[-2].target = y_pred2 - stepsize * dLdhL1
        for i in reversed(range(self.dim - 2)):
            self.layers[i].target = self.layers[i + 1].backward(self.layers[i + 1].target)

        # refinement
        repeat = 200

        import matplotlib.pyplot as plt
        x_plot = np.arange(repeat)
        ref_plot = np.zeros((self.dim - 2, repeat))
        tar_plot = np.zeros((self.dim - 2, repeat))
        h_plot = np.zeros((self.dim - 2, repeat))

        ret = True

        for num in range(repeat):
            for i in reversed(range(self.dim - 2)):
                gt = self.layers[i + 1].backward(self.layers[i + 1].target)
                ft_1 = self.layers[i + 1].forward(self.layers[i].target, update=False)
                gft_1 = self.layers[i + 1].backward(ft_1)
                self.layers[i].target += gt - gft_1
                # plot
                ref_plot[i, num] = np.linalg.norm(gt - gft_1)
                tar_plot[i, num] = np.linalg.norm(self.layers[i + 1].target - ft_1)
                h_plot[i, num] = np.linalg.norm(
                    self.layers[i].target - self.layers[i].linear_activation)
                if num == repeat - 1:
                    if abs(gt - gft_1).max() > 1:
                        #print(gt - gft_1)
                        ret = False

        if ret and False:
            index = 3
            t_i = self.layers[index].target
            t_i_1 = self.layers[index + 1].forward(t_i, update=False)
            print(np.linalg.norm(self.layers[index + 1].target -
                                 t_i_1) / (np.linalg.norm(t_i_1) + 1e-10))

        return ret

        """
        for i in range(self.dim - 2):
            plt.figure()
            plt.title(f"refinement_{i}")
            plt.plot(x_plot, ref_plot[i], label="refinement")
            plt.plot(x_plot, tar_plot[i], label="target")
            plt.legend()
            plt.xlabel("repeat")
            plt.savefig(f"refinement_{i}.png")

        for i in range(self.dim - 2):
            plt.figure()
            plt.title(f"refinement_{i}")
            plt.plot(x_plot, h_plot[i], label="hi-ti")
            plt.legend()
            plt.xlabel("repeat")
            plt.savefig(f"h_{i}.png")
        raise NotImplementedError
        """

    def compute_grad(self, x, y, y_pred):
        ret = [None] * self.dim
        global_loss = np.linalg.norm(self.layers[-1].target - self.layers[-1].linear_activation)**2
        for i in range(self.dim):
            local_loss = np.linalg.norm(self.layers[i].target - self.layers[i].linear_activation)**2
            lr = global_loss / (local_loss + 1e-12)
            h_previous = self.layers[i - 1].linear_activation if i != 0 else x
            s = self.layers[i].activation_function(h_previous)
            n = s / np.linalg.norm(s)**2

            self.lr_plot = np.append(self.lr_plot, lr)
            self.w_norm_plot = np.append(self.w_norm_plot,
                                         np.linalg.norm(lr * (self.layers[i].target -
                                                              self.layers[i].linear_activation).reshape(-1, 1)@n.reshape(1, -1)))
            ret[i] = lr * (self.layers[i].target -
                           self.layers[i].linear_activation).reshape(-1, 1)@n.reshape(1, -1)
        return ret

    def update_weights(self, x, y, y_pred):
        global_loss = np.linalg.norm(self.layers[-1].target - self.layers[-1].linear_activation)**2
        for i in range(self.dim):
            local_loss = np.linalg.norm(self.layers[i].target - self.layers[i].linear_activation)**2
            lr = global_loss / (local_loss + 1e-12)
            h_previous = self.layers[i - 1].linear_activation if i != 0 else x
            s = self.layers[i].activation_function(h_previous)
            n = s / np.linalg.norm(s)**2

            self.lr_plot = np.append(self.lr_plot, lr)
            self.w_norm_plot = np.append(self.w_norm_plot,
                                         np.linalg.norm(lr * (self.layers[i].target -
                                                              self.layers[i].linear_activation).reshape(-1, 1)@n.reshape(1, -1)))
            self.layers[i].weight += lr * (self.layers[i].target -
                                           self.layers[i].linear_activation).reshape(-1, 1)@n.reshape(1, -1)

    def update_weights_regular(self, x, y, y_pred):

        global_loss = ((self.layers[-2].target - self.layers[-2].linear_activation)**2).sum()
        for i in range(self.dim - 1):
            local_loss = ((self.layers[i].target - self.layers[i].linear_activation)**2).sum()
            lr = global_loss / (local_loss + 1e-12)
            h_previous = self.layers[i - 1].linear_activation if i != 0 else x
            s = self.layers[i].activation_function(h_previous)
            n = s / np.linalg.norm(s)**2
            self.lr_plot = np.append(self.lr_plot, lr)
            self.w_norm_plot = np.append(self.w_norm_plot,
                                         np.linalg.norm(lr * (self.layers[i].target -
                                                              self.layers[i].linear_activation).reshape(-1, 1)@n.reshape(1, -1)))
            self.layers[i].weight += lr * (self.layers[i].target -
                                           self.layers[i].linear_activation).reshape(-1, 1)@n.reshape(1, -1)
        # last layer
        global_loss = ((self.layers[-1].target - self.layers[-1].linear_activation)**2).sum()
        h_previous = self.layers[-2].linear_activation
        s = self.layers[-1].activation_function(h_previous)
        n = s / np.linalg.norm(s)**2
        self.w_norm_plot = np.append(self.w_norm_plot,
                                     np.linalg.norm((self.layers[-1].target -
                                                     self.layers[-1].linear_activation).reshape(-1, 1)@n.reshape(1, -1)))
        self.layers[-1].weight += (self.layers[-1].target -
                                   self.layers[-1].linear_activation).reshape(-1, 1)@n.reshape(1, -1)

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
                                activation_function=kwargs["activation_function"],
                                activation_derivative=kwargs["activation_derivative"])
        return layers

    def reconstruction_loss(self, x):
        y_pred = x
        for i in range(self.dim - 1):
            y_pred = self.layers[i].forward(y_pred, update=False)
        x_pred = y_pred
        for i in reversed(range(self.dim - 1)):
            x_pred = self.layers[i].backward(x_pred)
        return np.linalg.norm(x_pred - x)
