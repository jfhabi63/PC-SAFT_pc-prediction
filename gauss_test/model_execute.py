import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from data_extraction import *
from Dataset import *





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_kwargs = {"dtype": 'str',
                  "sep": ';',
                  "header": 0,
                  "float_precision": 'round_trip',
                  "na_values": 'none'}
raw_data = pd.read_csv("./active_model/inputs_test.csv",**dataset_kwargs,index_col=0)
raw_data = raw_data.astype('float32')
comp = list(raw_data.index)
X = raw_data.iloc[:,:].to_numpy()
y=[]
print(X,comp)


name = "model_gauss"
model_act = torch.load('./active_model/'+name)
model_act.eval()
dataset = Dataset_input(X,y,comp,train=False)
input_length = int(X.shape[1])
symm_mode = dataset.symm_mode
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=5, num_workers=0)


def calc_pred(data_loader, model):
    num_targets =3
    model.eval()
    mean = []
    std = []
    for Xi in data_loader:
        if symm_mode == 1:
            Xi = Xi.to(device)
        elif symm_mode == 2:
            Xi = Xi.to(device).reshape(-1, input_length)
        pred = model(Xi.to(device))  ### shape = (num_targets*2 für std_eval)
        pred_res = pred.cpu().detach().numpy()
        num_outputs = pred.size(1)

        if num_targets == num_outputs:
            mean.extend(pred_res.tolist())
        else:
            mean.extend(pred_res[:, 0:(num_outputs - 1)].tolist())
            std.extend(pred_res[:, -num_outputs:])
    mean = retransform(mean)
    abs_error = []
    diff_error = []
    if symm_mode == 2:
        for l in range(int(len(mean) / 2)):
            diff_error.append(abs(mean[2 * l] - mean[2 * l + 1]))
    return mean, std, abs_error, diff_error


with torch.no_grad():
    mean, std, error, diff_error = calc_pred(data_loader,model_act)
    y_true = mean
    comp = list(dataset.comp)
    df = pd.DataFrame(data=mean,columns=["m_seg","sigma","eps/k"],index=comp)
    print(df)
    df.to_csv(r'./active_model/ml_calc.csv', index=True, sep=';')







