import torch
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from Dataset import *
from model_BNN import *
from data_extraction import *


from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_fraction = 0.8
dataset = Dataset_input()

train_size = int(train_fraction * len(dataset))
test_size = len(dataset) - train_size
torch.manual_seed(1)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
test_dataset = val_dataset


loader_kwargs = {
    "batch_size": 64,
    "drop_last": False,
    "num_workers": 0,
    "pin_memory": False,
    "persistent_workers": False,
}
train_loader = DataLoader(dataset=train_dataset,
                          **loader_kwargs
                          )
val_loader = DataLoader(dataset=val_dataset,
                        **loader_kwargs
                        )
test_loader = DataLoader(dataset=test_dataset,
                         **loader_kwargs
                         )

model = BayesianModel().to(device)
optimizer = optim.Adam(model.parameters(),lr=0.01)
criterion = torch.nn.MSELoss()

iter = 0
epochs = 100
loss_history = {"train": [], "validation": []}

for i in range(epochs):
    if i % 5 == 0:
        print(f"Epoch: {i}/{epochs}")
    for batch, (X,y) in enumerate(train_loader):
        X = X.to(device).reshape(-1,28)
        y = y.to(device).reshape(-1,1)
        optimizer.zero_grad()

        loss = model.sample_elbo(inputs=X,
                                 labels= y,
                                 criterion= criterion,
                                 sample_nbr=5)
        if batch%10==0 and i % 5==0:
            print(f"loss: {loss.item()}")
        loss.backward()
        optimizer.step()


        iter+=1
print("Training finished!")

model.eval()

pred_train_mean, pred_train_std = evaluate_regression(model,train_loader)
#pred_test_mean, pred_test_std = evaluate_regression(model,test_loader)











