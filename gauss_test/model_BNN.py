import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch.nn.functional as F
import torchvision


device = "cuda" if torch.cuda.is_available() else "cpu"


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.lin1 = nn.Linear(28,32)
        self.lin2 = nn.Linear(32,32)
        self.lin3 = nn.Linear(32,1)

    def forward(self, x):
        forward = self.lin1(x)
        forward = F.tanh(forward)
        forward = self.lin2(forward)
        forward = F.tanh(forward)
        forward = self.lin3(forward)
        return forward

net = NN().to(device)

def model(x_data, y_data):
    lin1w_prior = dist.Normal(loc=torch.zeros_like(net.lin1.weight),
                         scale=torch.ones_like(net.lin1.weight))
    lin1b_prior = dist.Normal(loc=torch.zeros_like(net.lin1.bias),
                         scale=torch.ones_like(net.lin1.bias))

    lin2w_prior = dist.Normal(loc=torch.zeros_like(net.lin2.weight),
                         scale=torch.ones_like(net.lin2.weight))
    lin2b_prior = dist.Normal(loc=torch.zeros_like(net.lin2.bias),
                         scale=torch.ones_like(net.lin2.bias))

    lin3w_prior = dist.Normal(loc=torch.zeros_like(net.lin3.weight),
                         scale=torch.ones_like(net.lin3.weight))
    lin3b_prior = dist.Normal(loc=torch.zeros_like(net.lin3.bias),
                         scale=torch.ones_like(net.lin3.bias))

    priors = {'lin1.weight': lin1w_prior,
              'lin1.bias': lin1b_prior,
              'lin2.weight': lin2w_prior,
              'lin2.bias:':lin2b_prior,
              'lin3.weight':lin3w_prior,
              'lin3.bias':lin3b_prior}

    lifted_module = pyro.random_module("module",net,priors)
    lifted_reg_module = lifted_module()

    l_hat = log_softmax(lifted_reg_module(x_data))
    pyro.sample("obs", obs=y_data)


