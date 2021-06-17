import numpy as np
import matplotlib.pyplot as plt
from BPNN import *
from DTTPNN import *
from utils import *


def main():
    dim, in_dim, out_dim, hid_dim = 5, 5, 1, 3

    dataset = make_dataset(sigma=0.3)
    datax = dataset[0]
    dataset[0] = datax.reshape(-1, 1)**np.arange(0, in_dim)
    plt.plot(dataset[0][:, 1], dataset[1])
    plt.savefig("image/dataset.png")

    trainset = [dataset[0][1:-1:2], dataset[1][0:-1:2]]
    testset = [dataset[0][1:-1:2], dataset[1][1:-1:2]]

    dttpnn = DTTPNN(dim, in_dim, out_dim, hid_dim)
    bpnn = BPNN(dim, in_dim, out_dim, hid_dim)

    # train
    lr, epochs = 0.1, 100
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
    plt.figure()
    plt.plot(testset[0][:, 1], testset[1], label="theor")
    plt.plot(testset[0][:, 1], dttpnn_pred, label="DTTP")
    plt.plot(testset[0][:, 1], bpnn_pred, label="BP")
    plt.legend()
    plt.savefig("image/testset.png")


if __name__ == '__main__':
    main()
