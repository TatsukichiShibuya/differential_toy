from dttp_layer_torch import *
from net import *
from utils import *


class dttp_net_torch(net):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, dataset, epochs, stepsize, lrb):
        for e in range(epochs):
            for x, y in zip(dataset[0], dataset[1]):
                # train backward
                for i in range(10):
                    _x = x + torch.normal(0, 1, size=x.shape)
                    _y = self.forward(_x)
                    self.update_backweights(lrb)

                # compute target
                y_pred = self.forward(x)
                self.compute_target(y, y_pred, stepsize)

                # train forward
                self.update_weights()

            # predict
            pred = torch.zeros_like(dataset[1])
            for i, x in enumerate(dataset[0]):
                pred[i] = self.predict(x)
            print(f"epoch {e:<4}: {(torch.norm(pred-dataset[1])**2)/(2*len(dataset[0]))}")

    def update_backweights(self, lrb):
        for i in range(self.dim):
            self.layers[i].update_backweight(lrb)

    def compute_target(self, y, y_pred, stepsize):
        # initialize
        loss = torch.norm(y_pred - y)**2
        self.zero_grad()
        loss.backward()
        with torch.no_grad():
            for d in range(self.dim - 2, self.dim):
                self.layers[d].target = self.layers[d].linear_activation - \
                    stepsize * self.layers[d].linear_activation.grad
            for d in reversed(range(self.dim - 2)):
                self.layers[d].target = self.layers[d + 1].backward(self.layers[d + 1].target)

        # refinement
        for num in range(100):
            for i in reversed(range(self.dim - 2)):
                gt = self.layers[i + 1].backward(self.layers[i + 1].target)
                ft = self.layers[i + 1].forward(self.layers[i].target, update=False)
                gft = self.layers[i + 1].backward(ft)
                self.layers[i].target += gt - gft

    def update_weights(self):
        global_loss = torch.norm(self.layers[-2].target - self.layers[-2].linear_activation)**2
        for d in range(self.dim):
            local_loss = torch.norm(self.layers[d].target - self.layers[d].linear_activation)**2
            lr = global_loss / (local_loss + 1e-12) if (self.dim - d > 2) else 1
            n = self.layers[d].activation / torch.norm(self.layers[d].activation)**2
            with torch.no_grad():
                self.layers[d].weight += lr * (self.layers[d].target -
                                               self.layers[d].linear_activation).reshape(-1, 1)@n.reshape(1, -1)

    def zero_grad(self):
        for d in range(self.dim):
            if self.layers[d].linear_activation.grad is not None:
                self.layers[d].linear_activation.grad.zero_()
            if self.layers[d].weight.grad is not None:
                self.layers[d].weight.grad.zero_()
            if self.layers[d].backweight.grad is not None:
                self.layers[d].backweight.grad.zero_()

    def init_layers(self, **kwargs):
        layers = [None] * self.dim
        layers[0] = dttp_layer_torch(in_dim=kwargs["in_dim"],
                                     out_dim=kwargs["hid_dim"],
                                     activation_function=kwargs["activation_function"],
                                     activation_derivative=kwargs["activation_derivative"])
        for i in range(1, self.dim - 1):
            layers[i] = dttp_layer_torch(in_dim=kwargs["hid_dim"],
                                         out_dim=kwargs["hid_dim"],
                                         activation_function=kwargs["activation_function"],
                                         activation_derivative=kwargs["activation_derivative"])

        layers[-1] = dttp_layer_torch(in_dim=kwargs["hid_dim"],
                                      out_dim=kwargs["out_dim"],
                                      activation_function=kwargs["activation_function"],
                                      activation_derivative=kwargs["activation_derivative"])
        return layers
