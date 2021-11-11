import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = nn.Sequential(nn.Linear(28, 32),
                                  nn.Tanh(),
                                  nn.Dropout(0.1),
                                  nn.Linear(32, 64),
                                  nn.Tanh(),
                                  nn.Dropout(0.4),
                                  nn.Linear(64, 64),
                                  nn.Tanh(),
                                  nn.Dropout(0.2),
                                  nn.Linear(64, 16),
                                  nn.Tanh(),
                                  nn.Dropout(0.15),
                                  nn.Linear(16, 1)
                                  )

    def forward(self, x):
        forward = self.net(x)
        return forward

