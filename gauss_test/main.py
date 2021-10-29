import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from Dataset import *
from Model import *
from data_extraction import *

from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
build_dir = Path("build")
if not build_dir.exists():
    build_dir.mkdir()

train_fraction = 0.8
dataset = NoiseDataset()

train_size = int(train_fraction * len(dataset))
test_size = len(dataset) - train_size
torch.manual_seed(13)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
test_dataset = val_dataset

loader_kwargs = {
    "batch_size": 32,
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

model_path = build_dir / "nn.torch"
model = Model()
model.apply(weights_init_uniform_rule)
model.to(device)

if model_path.exists():
    model.load_state_dict(torch.load(model_path))
    model.eval()
else:
    loss_fn = torch.nn.GaussianNLLLoss(eps=1e-8)
    optimizer = torch.optim.Adam(model.parameters(),
                                 weight_decay=0,
                                 lr=0.0005
                                 )

    epochs = 8000
    start_var = 1

    loss_history = {"train": [], "validation": []}
    for t in range(epochs):
        print(
            f"\nEpoch {t + 1}/{epochs}"
            "\n-------------------------------"
            "\ntrain:"
        )
        if t < start_var:
            train(train_loader, model, loss_fn, optimizer, loss_history["train"], fit_variance=False)
        else:
            train(train_loader, model, loss_fn, optimizer, loss_history["train"], fit_variance=True)

        print("validate:")
        if t < start_var:
            validate(val_loader, model, loss_fn, loss_history["validation"], fit_variance=False)
        else:
            validate(val_loader, model, loss_fn, loss_history["validation"], fit_variance=True)

    visualize_training(
        loss_history["train"],
        loss_history["validation"],
        build_dir / "loss.png",
    )
    with open(build_dir / "loss.pickle", "wb") as f:
        pickle.dump(loss_history, f)
    #torch.save(model.state_dict(), model_path)
    torch.save(model,
               './build/model_gauss')

    print("\nTraining done!")


def calc_pred(data_loader, model):
    mean = []
    std = []
    y = []
    abs_error=[]
    for Xi, yi in data_loader:
        pred = model(Xi.to(device))
        pred_res = pred.cpu().detach().numpy()
        abs_error.extend(abs(pred_res[:, 0] - yi.cpu().detach().numpy()[:,0]))
        mean.extend(pred_res[:, 0])
        std.extend(pred_res[:, 1] ** 0.5)
        true_res = yi.cpu().detach().numpy()
        true_res = true_res.round(6)
        y.extend(true_res[:, 0].round(6))
    return mean,std,y, abs_error

pred_train_mean,pred_train_std,y_train, abs_error_train = calc_pred(train_loader,model)
pred_test_mean,pred_test_std,y_test, abs_error_test = calc_pred(test_loader,model)

print(pred_train_mean)
print(pred_train_std)
print(y_train)

print(pred_test_mean)
print(pred_test_std)
print(y_test)

result_export(pred_test_mean,
              pred_test_std,
              y_test,
              abs_error_test,
              'output_test'
              )
result_export(pred_train_mean,
              pred_train_std,
              y_train,
              abs_error_train,
              'output_train'
              )
plot_training(y_train,
              pred_train_mean,
              y_test,
              pred_test_mean,
              )
