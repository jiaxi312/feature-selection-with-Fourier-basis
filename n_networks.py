import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    """A simple network with only full connected linear layers"""

    def __init__(self, input_dim, output_dim, softmax=False):
        super(SimpleNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        self.softmax = nn.Softmax(dim=-1) if softmax else None

    def forward(self, x):
        x = self.layers(x)
        if self.softmax:
            x = self.softmax(x)
        return x
