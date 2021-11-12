import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(28, 32),
                                  nn.Tanh(),
                                  nn.Dropout(0.4),
                                  nn.Linear(32, 128),
                                  nn.Tanh(),
                                  nn.Dropout(0.4),
                                  nn.Linear(128, 256),
                                  nn.Tanh(),
                                  nn.Dropout(0.3),
                                  nn.Linear(256, 512),
                                  nn.Tanh(),
                                  nn.Dropout(0.4),
                                  nn.Linear(512, 512),
                                  nn.Tanh(),
                                  nn.Dropout(0.4),
                                  nn.Linear(512, 256),
                                  nn.Tanh(),
                                  nn.Dropout(0.4),
                                  nn.Linear(256, 128),
                                  nn.Tanh(),
                                  nn.Dropout(0.4),
                                  nn.Linear(128, 32),
                                  nn.Tanh(),
                                  nn.Dropout(0.4),
                                  nn.Linear(32, 16),
                                  nn.Tanh(),
                                  nn.Dropout(0.4),
                                  nn.Linear(16, 1)
                                  )
        self.net2 = nn.Sequential(nn.Linear(28, 32),
                                  nn.Tanh(),
                                  nn.Dropout(0),
                                  nn.Linear(32, 64),
                                  nn.Tanh(),
                                  nn.Dropout(0),
                                  nn.Linear(64, 128),
                                  nn.Tanh(),
                                  nn.Dropout(0),
                                  nn.Linear(128, 128),
                                  nn.Tanh(),
                                  nn.Dropout(0),
                                  nn.Linear(128, 128),
                                  nn.Tanh(),
                                  nn.Dropout(0),
                                  nn.Linear(128, 128),
                                  nn.Tanh(),
                                  nn.Dropout(0),
                                  nn.Linear(128, 64),
                                  nn.Tanh(),
                                  nn.Dropout(0),
                                  nn.Linear(64, 16),
                                  nn.Tanh(),
                                  nn.Dropout(0),
                                  nn.Linear(16, 1),
                                  )


    def forward(self, x, fit_variance=False):
        # torch.log(torch.var(x))
        #forward = torch.cat((self.net1(x), torch.ones(self.net1(x).size(0), 1) * 0.0018), dim=1)
        forward = torch.cat((self.net1(x), self.net2(x).abs()), dim=1)
        return forward



def train(dataloader, model, loss_fn, optimizer, loss_history=None, fit_variance=False):
    #size = len(dataloader.sampler)
    model.train()

    tmp_loss_history = []
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)


        # Compute prediction error
        pred = model(X).to(device)
        if fit_variance == False:
            loss = ((pred - y) ** 2).mean().to(device)
        else:
            #for n in model.net1.parameters():
            #    n.requires_grad=False
            loss = loss_fn(pred[:, 0], y[:,0], pred[:, 1]).to(device)
            MSE_loss = ((pred[:,0] - y[:,0]) ** 2).mean().to(device)
            if batch % 100 == 0:
                print(f'MSE_loss; {MSE_loss:.5f}')



        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(f"loss: {loss.item():>7f}")


        tmp_loss_history.append(loss.item())

    if loss_history is not None:
        loss_history.append(np.mean(tmp_loss_history))


def validate(dataloader, model, loss_fn, loss_history=None, fit_variance=False):
    #size = len(dataloader.sampler)
    model.eval()

    tmp_loss_history = []
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X)
        if fit_variance == False:
            loss = ((pred - y) ** 2).mean()
        else:
            loss = loss_fn(pred[:, 0], y[:,0], pred[:, 1].abs())

        if batch % 50 == 0:
            print(f"loss: {loss.item():.5f}")

        tmp_loss_history.append(loss.item())

    if loss_history is not None:
        loss_history.append(np.mean(tmp_loss_history))


def visualize_training(train_loss, test_loss, filename):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(len(train_loss)), train_loss, label="train")
    ax1.plot(range(len(test_loss)), test_loss, label="test")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    if filename:
        plt.savefig(filename)


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

