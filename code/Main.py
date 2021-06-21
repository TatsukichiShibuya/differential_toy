import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch import nn, optim
import torch

from bp_net import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=5)
    parser.add_argument("--in_dim", type=int, default=3)
    parser.add_argument("--hid_dim", type=int, default=3)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--activation_function", type=str, default="leakyrelu")
    parser.add_argument("--model", type=str, default="BP")
    parser.add_argument("--problem", type=str, default="regression")

    args = parser.parse_args()
    return args


def main(**kwargs):
    # make dataset
    if kwargs["problem"] == "regression":
        dataset = make_dataset_distance(dim=kwargs["in_dim"])
        trainset = [dataset[0][0:-1:2],
                    dataset[1][0:-1:2]]
        testset = [dataset[0][1:-1:2],
                   dataset[1][1:-1:2]]
        loss_function = (lambda x, y: mse(x, y))
        loss_derivative = (lambda x, y: mse_derivative(x, y))

    # initialize model
    if kwargs["activation_function"] == "leakyrelu":
        activation_function = (lambda x: leakyrelu(x, a=0.2))
        activation_derivative = (lambda x: leakyrelu_derivative(x, a=0.2))

    if kwargs["model"] == "BP":
        model = bp_net(dim=kwargs["dim"],
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

    # train
    model.train(trainset, kwargs["epochs"], kwargs["learning_rate"])

    # test
    pred = np.zeros_like(testset[1])
    for i, x in enumerate(testset[0]):
        pred[i] = model.predict(x)
    print(f"{kwargs['model']}: loss {np.sqrt((pred-testset[1])**2).sum()/2/testset[0].shape[0]}")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(-10, 10.1, 0.1)
    y = np.arange(-10, 10.1, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = model.predict(np.array([X[i, j], Y[i, j]]))
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5, linewidth=0.3)
    ax.view_init(elev=60, azim=60)
    fig.savefig(f"3dplot.png")


def debug(**kwargs):
    dataset = make_dataset_distance()
    trainset = [torch.from_numpy(dataset[0][0:-1:3].astype(np.float32)).clone(),
                torch.from_numpy(dataset[1][0:-1:3].astype(np.float32)).clone()]
    valset = [torch.from_numpy(dataset[0][0:-1:2].astype(np.float32)).clone(),
              torch.from_numpy(dataset[1][0:-1:2].astype(np.float32)).clone()]
    testset = [torch.from_numpy(dataset[0][1:-1:3].astype(np.float32)).clone(),
               torch.from_numpy(dataset[1][1:-1:3].astype(np.float32)).clone()]

    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(kwargs["in_dim"], kwargs["hid_dim"]),
                          nn.ReLU(),
                          nn.Linear(kwargs["hid_dim"], kwargs["out_dim"]))

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=kwargs["learning_rate"])

    for e in range(kwargs["epochs"]):
        for x, y in zip(trainset[0], trainset[1]):
            y_p = model(x)[0]
            loss = criterion(y_p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {e:<4}: ")

        pred = torch.zeros_like(trainset[1])
        for i, x in enumerate(trainset[0]):
            pred[i] = model(x)[0].item()
        print(f"\ttrains: {np.sqrt((pred-trainset[1])**2).sum()/trainset[0].shape[0]}")

        pred = torch.zeros_like(valset[1])
        for i, x in enumerate(valset[0]):
            pred[i] = model(x)[0].item()
        print(f"\tval  : {np.sqrt((pred-valset[1])**2).sum()/valset[0].shape[0]}")

    pred = torch.zeros_like(testset[1])
    for i, x in enumerate(testset[0]):
        pred[i] = model(x)[0].item()
    print(f"TORCH: loss {np.sqrt((pred-testset[1])**2).sum()/testset[0].shape[0]}")


if __name__ == '__main__':
    FLAGS = vars(get_args())
    print(FLAGS)
    main(**FLAGS)
    # debug(**FLAGS)
