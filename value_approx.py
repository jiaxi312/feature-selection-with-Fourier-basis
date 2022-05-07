import numpy as np
import torch
import torch.nn as nn
from n_networks import SimpleNet


class ValueApproximation:
    def __init__(self):
        pass

    def __call__(self, s) -> int:
        """Returns the approximated value of given state s"""
        pass

    def update(self, *args, **kwargs):
        """Updates the weight"""
        pass


class ValueApproximationWithNN(ValueApproximation):
    def __init__(self, num_states, alpha):
        super(ValueApproximationWithNN, self).__init__()
        self.num_states = num_states
        self.alpha = alpha

        self.net = SimpleNet(num_states, 1, softmax=False)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.alpha, betas=(0.9, 0.999))

    def __call__(self, s) -> float:
        self.net.eval()
        with torch.no_grad():
            s = torch.tensor(s.flatten(), dtype=torch.float)
            value = self.net(s)
            return value.item()

    def update(self, s, G):
        self.net.eval()
        s = torch.tensor(s.flatten(), dtype=torch.float)
        prediction = self.net(s)
        actual = torch.tensor([G], dtype=torch.float)

        loss = self.criterion(prediction, actual)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ValueApproximationWithFourier(ValueApproximation):
    def __init__(self, num_states, order=1, weight_values=None, c_values=None):
        super(ValueApproximationWithFourier, self).__init__()

        self.num_states = num_states
        self.order = order
        self.num_features = (order + 1) ** num_states
        self.criterion = nn.MSELoss()
        self.c = self._init_c(c_values)
        self.w = self._init_weights(weight_values)

    def _init_weights(self, weight_values):
        w = torch.zeros(1, self.num_features, dtype=torch.float, requires_grad=True)
        if weight_values is not None:
            w = torch.tensor(weight_values, dtype=torch.float, requires_grad=True)
        return w

    def _init_c(self, c_values):
        c = torch.randint(0, self.order + 1, (self.num_features, self.num_states),
                          dtype=torch.float, requires_grad=False)
        if c_values is not None:
            c = torch.tensor(c_values,
                             dtype=torch.float,
                             requires_grad=False)
            c = c.view(self.num_features, self.num_states)
        return c

    def __call__(self, s) -> float:
        with torch.no_grad():
            s = torch.tensor([s.flatten()], dtype=torch.float)
            cos = torch.cos(self.c.matmul(s.t()))
            v = self.w.matmul(cos)
            return v.item()

    def __str__(self):
        states_str = ''.join(['%12d' % i for i in range(self.num_states)] + ['\t  Weights'])
        c_str = []
        for r in range(self.c.shape[0]):
            c_str.append('C:')
            for c in range(self.c.shape[1]):
                c_str.append('\t\t%4.4f' % self.c[r, c].item())
            c_str.append(f'\t\t{self.w[r].detach().numpy()}')
            c_str.append('\n')
        c_str = ''.join(c_str)
        return states_str + '\n' + c_str


if __name__ == '__main__':
    a = [i for i in range(32)]
    b = [i for i in range(24)]
    V = ValueApproximationWithFourier(3, 0.1, weight_values=a, c_values=b)
    print(V)
