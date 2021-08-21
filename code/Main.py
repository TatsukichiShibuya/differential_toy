from utils import *
from dttp_net_debug import *
from bp_net import *
from dctp_net import *
from dttp_net_torch import *

import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch import nn, optim
import pickle
import torch
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BP")
    parser.add_argument("--problem", type=str, default="regression")
    # parameters
    parser.add_argument("--size",    type=int, default=500)
    parser.add_argument("--dim",     type=int, default=5)
    parser.add_argument("--in_dim",  type=int, default=2)
    parser.add_argument("--hid_dim", type=int, default=3)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--epochs",  type=int, default=1000)
    parser.add_argument("--activation_function", type=str, default="leakyrelu")
    # parameters used in BP
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-6)
    # parameters used in DTTP and DCTP
    parser.add_argument("--stepsize", type=float, default=2e-5)
    parser.add_argument("--learning_rate_for_backward", "-lrb", type=float, default=1e-2)
    args = parser.parse_args()
    return args


def main(**kwargs):
    # make dataset
    if kwargs["problem"] == "regression":
        if kwargs["model"] in ["DCTP", "DTTP_torch"]:
            dataset = make_dataset_distance(size=kwargs["size"],
                                            dim=kwargs["in_dim"],
                                            use_torch=True)
        else:
            dataset = make_dataset_distance(size=kwargs["size"],
                                            dim=kwargs["in_dim"])
        trainset = [dataset[0][0:-1:2],
                    dataset[1][0:-1:2]]
        testset = [dataset[0][1:-1:2],
                   dataset[1][1:-1:2]]
        loss_function = (lambda x, y: mse(x, y))
        loss_derivative = (lambda x, y: mse_derivative(x, y))
    else:
        sys.tracebacklimit = 0
        raise NotImplementedError("No such dataset")

    # initialize model
    if kwargs["activation_function"] == "leakyrelu":
        if kwargs["model"] in ["DCTP", "DTTP_torch"]:
            activation_function = torch.nn.LeakyReLU(0.2)
            activation_derivative = None
        else:
            activation_function = (lambda x: leakyrelu(x, a=0.2))
            activation_derivative = (lambda x: leakyrelu_derivative(x, a=0.2))
    else:
        sys.tracebacklimit = 0
        raise NotImplementedError("No such activation")

    if kwargs["model"] == "BP":
        model = bp_net(dim=kwargs["dim"],
                       in_dim=kwargs["in_dim"],
                       out_dim=kwargs["out_dim"],
                       hid_dim=kwargs["hid_dim"],
                       activation_function=activation_function,
                       activation_derivative=activation_derivative,
                       loss_function=loss_function,
                       loss_derivative=loss_derivative)
    elif kwargs["model"] == "DCTP":
        model = dctp_net(dim=kwargs["dim"],
                         in_dim=kwargs["in_dim"],
                         out_dim=kwargs["out_dim"],
                         hid_dim=kwargs["hid_dim"],
                         activation_function=activation_function,
                         activation_derivative=activation_derivative,
                         loss_function=loss_function,
                         loss_derivative=loss_derivative)
    elif kwargs["model"] == "DTTP":
        model = dttp_net(dim=kwargs["dim"],
                         in_dim=kwargs["in_dim"],
                         out_dim=kwargs["out_dim"],
                         hid_dim=kwargs["hid_dim"],
                         activation_function=activation_function,
                         activation_derivative=activation_derivative,
                         loss_function=loss_function,
                         loss_derivative=loss_derivative)
    elif kwargs["model"] == "DTTPflozen":
        model = dttp_net(dim=kwargs["dim"],
                         in_dim=kwargs["in_dim"],
                         out_dim=kwargs["out_dim"],
                         hid_dim=kwargs["hid_dim"],
                         activation_function=activation_function,
                         activation_derivative=activation_derivative,
                         loss_function=loss_function,
                         loss_derivative=loss_derivative)
        with open('model.pickle', 'rb') as f:
            layers = pickle.load(f)
        for i in range(kwargs["dim"]):
            model.layers[i].weight = layers[i]
    elif kwargs["model"] == "DTTP_torch":
        model = dttp_net_torch(dim=kwargs["dim"],
                               in_dim=kwargs["in_dim"],
                               out_dim=kwargs["out_dim"],
                               hid_dim=kwargs["hid_dim"],
                               activation_function=activation_function,
                               activation_derivative=activation_derivative,
                               loss_function=loss_function,
                               loss_derivative=loss_derivative)
    else:
        sys.tracebacklimit = 0
        raise NotImplementedError("No such model")

    # train
    if kwargs["model"] == "BP":
        model.train(trainset, kwargs["epochs"], kwargs["learning_rate"])
    elif kwargs["model"] == "DCTP":
        model.train(trainset, kwargs["epochs"], kwargs["stepsize"],
                    kwargs["learning_rate_for_backward"])
    elif kwargs["model"] in ["DTTP", "DTTPflozen"]:
        model.train_last_full(trainset, kwargs["epochs"], kwargs["stepsize"],
                              kwargs["learning_rate_for_backward"])
    elif kwargs["model"] == "DTTP_torch":
        model.train(trainset, kwargs["epochs"], kwargs["stepsize"],
                    kwargs["learning_rate_for_backward"])

    # test
    if kwargs["model"] in ["DCTP", "DTTP_torch"]:
        pred = torch.zeros_like(testset[1])
        for i, x in enumerate(testset[0]):
            pred[i] = model.predict(x)
        print(f"{kwargs['model']}: loss {(torch.norm(pred-testset[1])**2)/(2*len(testset[0]))}")
    else:
        pred = np.zeros_like(testset[1])
        for i, x in enumerate(testset[0]):
            pred[i] = model.predict(x)
        print(f"{kwargs['model']}: loss {(np.linalg.norm(pred-testset[1])**2)/(2*len(testset[0]))}")

    # plot
    if kwargs["in_dim"] == 2:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = np.arange(-10, 10.1, 0.1)
        y = np.arange(-10, 10.1, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if kwargs["model"] in ["DCTP", "DTTP_torch"]:
                    Z[i, j] = model.predict(torch.tensor([X[i, j], Y[i, j]], dtype=torch.float))
                else:
                    Z[i, j] = model.predict(np.array([X[i, j], Y[i, j]]))
        ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, linewidth=0.3)
        ax.view_init(elev=60, azim=60)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.savefig(f"3dplot_{kwargs['model']}.png")


if __name__ == '__main__':
    FLAGS = vars(get_args())
    print(FLAGS)
    main(**FLAGS)
