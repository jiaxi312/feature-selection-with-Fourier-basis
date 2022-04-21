import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from n_networks import SimpleNet


class PiApproximation:
    def __init__(self):
        pass

    def __call__(self, s):
        """Returns the pi(.|S), the selected action given state s"""
        pass

    def update(self, *args, **kwargs):
        """Updates the weights"""
        pass


class PiApproximationWithNN(PiApproximation):
    def __init__(self, num_states, num_actions, alpha):
        super(PiApproximationWithNN, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions,

        # neural net parameters
        self.alpha = alpha
        self.net = SimpleNet(num_states, num_actions, softmax=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.alpha, betas=(0.9, 0.999))

    def __call__(self, s) -> [int, float]:
        s = torch.tensor(s.flatten(), dtype=torch.float)
        actions_prob = self.net(s)
        m = Categorical(actions_prob)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update(self, log_prob, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.net.train()
        loss = -log_prob * gamma_t * delta
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PiApproximationWithFourier(PiApproximation):
    def __init__(self, num_states, num_actions, alpha, order=1, c_values=None, weight_values=None):
        super(PiApproximationWithFourier, self).__init__()

        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.num_features = (order + 1) ** num_states
        self.order = order

        self.c = self._init_c(c_values)
        self.w = self._init_w(weight_values)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD([self.w], lr=self.alpha)
        self.softmax = nn.Softmax(dim=0)

    def _init_c(self, c_values):
        c = torch.randint(0, self.order + 1, (self.num_features, self.num_states),
                          dtype=torch.float, requires_grad=False)
        if c_values is not None:
            c = torch.tensor(c_values,
                             dtype=torch.float,
                             requires_grad=False)
            c = c.view(self.num_features, self.num_states)
        return c

    def _init_w(self, weight_values):
        w = torch.zeros(self.num_actions, self.num_features, dtype=torch.float, requires_grad=True)
        if weight_values is not None:
            w = torch.tensor(weight_values, dtype=torch.float, requires_grad=False).view(self.num_actions, self.num_features)
            w = torch.tensor(w.detach(), requires_grad=True)
        return w

    def __call__(self, s) -> [int, float]:
        s = torch.tensor([s.flatten()], dtype=torch.float)
        cos = torch.cos(self.c.matmul(s.t()))
        action_probs = self.softmax(self.w.matmul(cos).squeeze())
        m = Categorical(action_probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update(self, log_prob, gamma_t, delta):
        loss = -log_prob * gamma_t * delta
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
