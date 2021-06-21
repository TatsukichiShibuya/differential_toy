import numpy as np
import matplotlib.pyplot as plt
from BPNN import *
from DTTPNN import *
from utils import *
from torch import nn, optim
import torch


def main(dim, indim, out_dim, hid_dim, lr, epochs):
    dataset = make_dataset_distance()
    trainset = [dataset[0][0:-1:2],
                dataset[1][0:-1:2]]
    testset = [dataset[0][1:-1:2],
               dataset[1][1:-1:2]]

    dttpnn = DTTPNN(dim, in_dim, out_dim, hid_dim)
    bpnn = BPNN(dim, in_dim, out_dim, hid_dim)

    # train
    dttpnn.train(trainset, epochs, lr)
    bpnn.train(dataset, epochs, lr)

    # test
    dttpnn_pred = np.zeros_like(testset[1])
    bpnn_pred = np.zeros_like(testset[1])
    for i, (x, y) in enumerate(zip(testset[0], testset[1])):
        # dttp
        pred = dttpnn.test(x)
        dttpnn_pred[i] = pred
        # bp
        pred = bpnn.test(x)
        bpnn_pred[i] = pred
    print(f"DTTP : loss {np.sqrt((dttpnn_pred-testset[1])**2).sum()}")
    print(f"BP : loss {np.sqrt((bpnn_pred-testset[1])**2).sum()}")


def debug(dim, in_dim, out_dim, hid_dim, lr, epochs):
    dataset = make_dataset_distance()
    trainset = [torch.from_numpy(dataset[0][0:-1:2].astype(np.float32)).clone(),
                torch.from_numpy(dataset[1][0:-1:2].astype(np.float32)).clone()]
    testset = [torch.from_numpy(dataset[0][1:-1:2].astype(np.float32)).clone(),
               torch.from_numpy(dataset[1][1:-1:2].astype(np.float32)).clone()]

    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(in_dim, hid_dim),
                          nn.ReLU(),
                          nn.Linear(hid_dim, out_dim))

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        for x, y in zip(trainset[0], trainset[1]):
            y_p = model(x)[0]
            loss = criterion(y_p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch {e:<4}: {loss.item()}")

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    dim, in_dim, out_dim, hid_dim = 5, 3, 1, 3
    lr, epochs = 2e-5, 10000
    # main(dim, indim, out_dim, hid_dim, lr, epochs)
    debug(dim, in_dim, out_dim, hid_dim, lr, epochs)
