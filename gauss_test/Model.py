import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
from torchsummary import summary
from Dataset import retransform

class Model(nn.Module):
    def __init__(self):
        from Dataset import symm_mode, bin_mode, num_targets, n_samples, input_length
        super(Model, self).__init__()
        act_fct=nn.ELU()
        self.net1 = nn.Sequential(nn.Linear(input_length, 2048),
                                  act_fct,
                                  nn.Dropout(0.11),
                                  nn.Linear(2048, 4096),
                                  act_fct,
                                  nn.Dropout(0.11),
                                  nn.Linear(4096, 1024),
                                  act_fct,
                                  nn.Dropout(0.11),
                                  nn.Linear(1024, 512),
                                  act_fct,
                                  nn.Dropout(0.11),
                                  nn.Linear(512, 32),
                                  act_fct,
                                  nn.Dropout(0.045),
                                  nn.Linear(32, num_targets)
                                  )
        self.net2 = nn.Sequential(nn.Linear(input_length, 32),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0),
                                  nn.Linear(32, 64),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0),
                                  nn.Linear(64, 128),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0),
                                  nn.Linear(128, 256),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0),
                                  nn.Linear(256, 128),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0),
                                  nn.Linear(128, 64),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0),
                                  nn.Linear(64, 32),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0),
                                  nn.Linear(32, 16),
                                  nn.LeakyReLU(),
                                  nn.Dropout(0),
                                  nn.Linear(16, num_targets),
                                  )



    def forward(self, x):
        from Dataset import std_eval
        # torch.log(torch.var(x))
        #forward = torch.cat((self.net1(x), torch.ones(self.net1(x).size(0), 1)), dim=1)
        if std_eval == True:
            forward = torch.cat((self.net1(x), self.net2(x).abs()), dim=1) ###symmetric, with sigma
        else:
            forward = self.net1(x)
        return forward



def train(dataloader, model, loss_fn, optimizer,scale_param, model_pLV,model_rho,scaler_loss,loss_history=None, fit_variance=False):
    #size = len(dataloader.sampler)
    model.train()
    from Dataset import input_length
    tmp_loss_history = []
    for batch, (X, y) in enumerate(dataloader):
        if X.size(1)==input_length:
            X = X.to(device)
            y = y.to(device)
        else:
            X = X.to(device).reshape(-1,input_length)
            y = y.to(device).reshape(-1,1)
        if fit_variance == False:
            pred = model(X).to(device)


            loss_rho,loss_pLV = loss_PCSAFT(y,pred,model_pLV,model_rho,scaler_loss,scale_param)

            for n in model.net2.parameters():
                n.requires_grad=False
            criterion=torch.nn.HuberLoss()  ##pc_param
            #diff =y[:,0]-pred[:,0]
            loss =criterion(pred,y).to(device)

            t1 = 5e-4*2
            t2 = 6e-4*1.4
            t3 = 0
            print(f"HL: {loss*t3:.4f} ; rho_loss: {loss_rho*t1:.4f} ; pLV_loss: {loss_pLV*t2:.4f}")
            loss = (loss*t3 + t1*loss_rho + t2*loss_pLV)/3

            #loss=(criterion(pred[:,0],y[:,0])+criterion(pred[:,1],y[:,1]+diff*t)+criterion(pred[:,2],y[:,2]+diff*t)).to(device) ##pc_param
            #loss = (((pred[:,0] - y[:,0]) ** 2).mean()*j)+((pred[::2,0]-pred[1::2,0])**2).mean()*k).to(device) ###auskommentieren bei symmetrischen Daten
            MSE_loss = ((pred[:, 0] - y[:,0]) ** 2).mean().to(device)
            if batch % 100 == 0:
                print(f'MSE_loss; {MSE_loss:.5f}') #symmetric
                #print(f'MSE_loss; {loss:.5f}')
        else:
            pred = model(X).to(device)
            for n in model.net2.parameters():
               n.requires_grad=True
            for n in model.net1.parameters():
               n.requires_grad=False
            loss = loss_fn(pred[:, 0],y[:,0], pred[:, 1]).to(device)
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


def validate(dataloader, model, loss_fn,model_pLV, model_rho, scaler_loss, scale_param,loss_history=None, fit_variance=False):
    #size = len(dataloader.sampler)
    model.eval()
    from Dataset import input_length
    loss_gesamt = []
    tmp_loss_history = []
    #pred_retr_full = np.empty([0,3])
    for batch, (X, y) in enumerate(dataloader):
        if X.size(1) == input_length:
            X = X.to(device)
            y = y.to(device)
        else:
            X = X.to(device).reshape(-1, input_length)
            y = y.to(device).reshape(-1, 1)

        # Compute prediction error
        pred = model(X)
        ##########retransform of predicted parameters######################
        loss_rho, loss_pLV = loss_PCSAFT(y, pred, model_pLV, model_rho, scaler_loss, scale_param)
        print(f"val_loss: {loss_rho:.1f} (rho) ; {loss_pLV:.1f} (pLV)")
        #############################################################


        if fit_variance == False:
            #loss = ((pred[:,0] - y[:,0]) ** 2).mean()
            criterion=torch.nn.HuberLoss()  ##pc_param
            loss=criterion(pred,y).to(device)

            t1 = 5e-4*2
            t2 = 6e-4*1.4
            t3 = 0
            loss = (loss*t3 + t1*loss_rho + t2*loss_pLV)/3

            loss_gesamt.append(loss.item())
        else:
            loss = loss_fn(pred[:, 0], y[:,0], pred[:, 1].abs())
            criterion=torch.nn.HuberLoss()  ##pc_param
            loss=criterion(pred,y).to(device)

        #if batch % 50 == 0:
            #print(f"loss: {loss.item():.5f}")
            #print(f"loss: {np.mean(loss_gesamt):.5f}")

        tmp_loss_history.append(loss.item())


    if loss_history is not None:
        loss_history.append(np.mean(tmp_loss_history))
    return np.mean(loss_gesamt)


def loss_PCSAFT(y,pred,model_pLV,model_rho,scaler_loss,scale_param):
    pred_retrans = (pred * (scale_param[2, :] - scale_param[3, :]) + scale_param[3, :]) * scale_param[1,:] + scale_param[0, :]
    y_retrans = ((y * (scale_param[2, :] - scale_param[3, :]) + scale_param[3, :]) * scale_param[1, :] + scale_param[0,:]).float()
    loss_input = torch.cat((y_retrans, pred_retrans), dim=1).float()
    loss_pLV_input = (loss_input - scaler_loss[2, 0:6]) / scaler_loss[3, 0:6]
    loss_rho_input = (loss_input - scaler_loss[0, 0:6]) / scaler_loss[1, 0:6]
    loss_pLV = (model_pLV(loss_pLV_input.float()) * scaler_loss[3, 6:] + scaler_loss[2, 6:]).abs()[:,1]
    loss_rho = (model_rho(loss_rho_input.float()) * scaler_loss[1, 6:] + scaler_loss[0, 6:]).abs()[:,0]
    return torch.mean(loss_rho),torch.mean(loss_pLV)



def visualize_training(train_loss, test_loss, filename):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(len(train_loss)), train_loss, label="train")
    ax1.plot(range(len(test_loss)), test_loss, label="test")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.axis([0,len(train_loss),0,0.01])
    ax1.legend()

    if filename:
        plt.savefig(filename)
    plt.close()


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

class loss_Model(nn.Module):
    def __init__(self):
        num_targets = 2
        input_length= 6
        super(loss_Model, self).__init__()
        act_fct=nn.LeakyReLU()
        self.net1 = nn.Sequential(nn.Linear(input_length, 512),
                                  act_fct,
                                  nn.Dropout(0),
                                  nn.Linear(512, 1024),
                                  act_fct,
                                  nn.Dropout(0),
                                  nn.Linear(1024, 2048),
                                  act_fct,
                                  nn.Dropout(0),
                                  nn.Linear(2048, 1024),
                                  act_fct,
                                  nn.Dropout(0),
                                  nn.Linear(1024, num_targets),
                                  )
    def forward(self, x):
        forward = self.net1(x)
        return forward