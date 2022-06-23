import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())

@variational_estimator
class BayesianModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(BayesianLinear(28, 32),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0),
                                  BayesianLinear(32, 128),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0),
                                 BayesianLinear(128, 256),
                                 nn.LeakyReLU(),
                                 nn.Dropout(0),
                                 BayesianLinear(256, 512),
                                 nn.LeakyReLU(),
                                 nn.Dropout(0),
                                 BayesianLinear(512, 512),
                                 nn.LeakyReLU(),
                                 nn.Dropout(0),
                                 BayesianLinear(512, 128),
                                 nn.LeakyReLU(),
                                 nn.Dropout(0),
                                 BayesianLinear(128, 64),
                                 nn.LeakyReLU(),
                                 nn.Dropout(0),
                                 BayesianLinear(64, 32),
                                 nn.LeakyReLU(),
                                 nn.Dropout(0),
                                  BayesianLinear(32, 16),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0),
                                  BayesianLinear(16, 1))
    def forward(self,x):
        return self.net(x)

def evaluate_regression(regressor,
                        dataloader,
                        samples=10,
                        std_multiplier=2):
    for Xi, yi in dataloader:
        Xi = Xi.to(device).reshape(-1, 28)
        yi = yi.to(device).reshape(-1, 1)
        print(Xi,yi)
        preds = [regressor(Xi) for i in range(samples)]
        preds = torch.stack(preds)
        means = preds.mean(axis=0)
        stds = preds.std(axis=0)
        y_upper = (means + std_multiplier * stds)
        y_lower = (means - std_multiplier * stds)
        ic_acc = ((y_lower <= yi) * (y_upper >= yi))
        ic_acc = ic_acc.float().mean()
        print(means, stds)

    return preds, stds #,ic_acc, (y_upper >= y).float().mean(), (y_lower <= y).float().mean()


