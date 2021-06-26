import numpy as np


def leakyrelu(x, a):
    x_copy = x.copy()
    frag = x < 0
    x_copy[frag] = a * x_copy[frag]
    return x_copy


def leakyrelu_derivative(x, a):
    x_copy = x.copy()
    frag = x < 0
    x_copy[frag] = a
    x_copy[~frag] = 1
    return x_copy


def mse(x, y):
    return ((x - y)**2).sum() / 2


def mse_derivative(x, y):
    return x - y


def make_dataset(sigma=1, seed=1):
    # f(x) = sin(x)+0.5x
    np.random.seed(seed)
    x = np.arange(0, 10.1, 0.1)
    y = np.sin(x) + x / 2 + np.random.normal(0, sigma, x.shape)
    return [x, y]


def make_dataset_distance(dim=3, seed=1):
    # dim次元空間での原点からの距離
    np.random.seed(seed)
    x = np.random.randn(500, dim) * 10
    y = np.sqrt((x**2).sum(axis=1))
    return [x, y]


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
