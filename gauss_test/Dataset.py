from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def load_raw_data(rand_seed,train=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global symm_mode, bin_mode, num_targets, std_eval
    global import_data

    symm_mode = 1  # non-symmetric(1), symmetric (2)
    bin_mode = 1  # purecomp(1),binary(2)
    std_eval = False  # activate 2nd NN
    num_targets = 3
    dataset_kwargs = {"dtype": 'str',
                      "sep": ';',
                      "header": 0,
                      "float_precision": 'round_trip',
                      "na_values": 'none'}
    import_data = 'purecomp_ECFP16384_5_full(1603)'#'purecomp_ECFP16384_5'#'purecomp_ECFP16384_5_full(0903)'  # 'cassens_test_symm'
    if train == True:
        folder = "input"
    else:
        folder = "active_model"
    if bin_mode == 1:
        raw = pd.read_csv(f'./{folder}/inputs_' + import_data + '.csv', **dataset_kwargs, index_col=0
                          )
        ECFP_cols = raw.columns[:-3]
        #raw_train = pd.read_csv(f'./{folder}/inputs_' + import_data + '_train.csv', **dataset_kwargs, index_col=0
        #                  )
        #raw_test = pd.read_csv(f'./{folder}/inputs_' + import_data + '_test.csv', **dataset_kwargs, index_col=0
        #                  )
        #ECFP_cols = raw_train.columns[:-3]


    else:
        raw = pd.read_csv(f'./{folder}/inputs_' + import_data + '.csv', **dataset_kwargs, index_col=[0, 1]
                          )
    for key in raw.columns:
        mask = raw[key].isna()
        if sum(mask) > 0:
            print(f"{key} : {sum(mask)}")
    raw.fillna(0, inplace=True)
    raw = raw.astype('float32')
    #raw_test = raw_test.astype('float32')     using defined test and training set
    #raw_train = raw_train.astype('float32')
    if train == True:
        y = raw.iloc[:, -symm_mode * num_targets:]
        X = raw.iloc[:, 0:(-symm_mode * num_targets)].to_numpy(dtype=np.float32)
        #y_test = raw_test.iloc[:, -symm_mode * num_targets:]
        #X_test = raw_test.iloc[:, 0:(-symm_mode * num_targets)].to_numpy(dtype=np.float32) #using defined test and training set
        #y_train = raw_train.iloc[:, -symm_mode * num_targets:]
        #X_train = raw_train.iloc[:, 0:(-symm_mode * num_targets)].to_numpy(dtype=np.float32)

    else:
        X = raw.iloc[:, :].to_numpy(dtype=np.float32)
    global n_samples, input_length
    n_samples = X.shape[0] * symm_mode
    comp = raw.index
    #n_samples = X_test.shape[0]+X_train.shape[0] #using defined test and training set
    #comp_test = raw_test.index   #using defined test and training set
    #comp_train = raw_train.index   #using defined test and training set

    #######################k-fold CV#####################################################
    kf = KFold(n_splits=5, random_state=rand_seed, shuffle=True)
    #print(kf.split(X))
    ##################################################################################

    X_train,X_test,y_train,y_test,comp_train,comp_test = train_test_split(X,y,comp,test_size=0.2,random_state=rand_seed)
    y_test_pretransform= y_test.to_numpy()
    y_train_pretransform = y_train.to_numpy()
    X_train_new, X_test_new = split_data(X_train,X_test,comp_train,comp_test,ECFP_cols)
    y_mean = [np.mean(y_train.iloc[:,i]) for i in range(y_train.shape[1])]
    y_std = [np.std(y_train.iloc[:,i]) for i in range(y_train.shape[1])]
    if train == True:
        scx = StandardScaler()
        scy = StandardScaler()
        mmx = MinMaxScaler()
        mmy = MinMaxScaler()
        X_train_new = scx.fit_transform(X_train_new)
        X_train_new = mmx.fit_transform(X_train_new)
        y_train = scy.fit_transform(y_train)
        y_max = [max(y_train[:,i]) for i in range(y_train.shape[1])]
        y_min = [min(y_train[:,i]) for i in range(y_train.shape[1])]
        y_train = mmy.fit_transform(y_train)
        X_test_new = scx.transform(X_test_new)
        X_test_new = mmx.transform(X_test_new)
        y_test = scy.transform(y_test)
        y_test = mmy.transform(y_test)

        scaling_parameters = torch.tensor([y_mean]+[y_std]+[y_max]+[y_min]).to(device)
        #    y=np.round(scy.inverse_transform(y),6)
        joblib.dump(scx, 'std_scaler_X.bin', compress=True)
        joblib.dump(scy, 'std_scaler_y.bin', compress=True)
        joblib.dump(mmx, 'minmax_scaler_X.bin', compress=True)
        joblib.dump(mmy, 'minmax_scaler_y.bin', compress=True)
    else:
        sc_X = joblib.load('./active_model/std_scaler_X.bin')
        minmax_X = joblib.load('./active_model/minmax_scaler_X.bin')
        X = sc_X.transform(X)
        X = minmax_X.transform(X)

    input_length = int(X_train_new.shape[1] / symm_mode)
    feature_loss = X_train.shape[1] - X_train_new.shape[1]
    return X_train_new,X_test_new,y_train,y_test,comp_train,comp_test, feature_loss, scaling_parameters, y_test_pretransform, y_train_pretransform






class Dataset_input(Dataset):
    def __init__(self,X,y, comp, transform=None,train=True):
        # data loading
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transform

        self.x = torch.from_numpy(X).to(device)
        if train == True:
            self.y = torch.from_numpy(y).to(device)  #reshape(-1, 1)   .to_numpy(dtype=np.float32)
        self.n_samples = X.shape[0]
        self.comp = comp
        self.input_length = input_length
        self.symm_mode = symm_mode

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x=self.transform(x)
        try:
            return x, y
        except AttributeError:
            return x

    def __len__(self):
        self.size = len(self.x)
        return self.size




def retransform(y):
    scy = joblib.load('std_scaler_y.bin')
    mmy = joblib.load('minmax_scaler_y.bin')
    y = pd.DataFrame(y)
    #y.to_csv(f"./output/data{len(y)}.csv",sep=";")
    if symm_mode==1:
        y = np.round(mmy.inverse_transform(y), 6)
        y = np.round(scy.inverse_transform(y), 6)


    elif symm_mode == 2:
        y = pd.concat([y, y], axis=1)
        y = np.round(scy.inverse_transform(y), 6)
        # y = y[:,0].reshape(1,len(y)).tolist()
        y = [num for elem in y for num in elem]
        del y[::2]
    return y


def split_data(X_train,X_test,comp_train,comp_test, ECFP_cols):
    total_hits=int(round(sum(sum(X_train[:,12:])+sum(X_test[:,12:])),0))
    size = X_train.shape[1]
    X_train_slice = X_train
    X_test_slice = X_test
    total_hits_per_comp = [round(sum(row[12:]),0) for row in X_test]
    count = 0
    lost_columns = []
    lost_data = pd.DataFrame(index=comp_test,dtype='float32')
    for i in range(size):
        if sum(X_train[:,i])==0 and i>11:
            lost_columns.append(ECFP_cols[i])
            X_train_slice = np.delete(X_train_slice,i - count,1)
            X_test_slice = np.delete(X_test_slice, i - count, 1)
            count += 1
            lost_data[f"{ECFP_cols[i]}"] = X_test[:, i]

    lost_data["hits_drop_per_comp"] = [sum(lost_data.loc[comp]) for comp in lost_data.index]
    hits_lost_per_sub =[sum(lost_data[col]) for col in lost_data.columns]
    lost_data = lost_data.append(pd.DataFrame(data=hits_lost_per_sub,
                                              index=lost_data.columns,
                                              columns=["hits_dropped_per_substructure"]).transpose(),
                                 ignore_index=False)

    loss_perc_per_comp = [round(a/b*100,1) for a,b in zip([sum(lost_data.loc[comp]) for comp in lost_data.index],total_hits_per_comp)]
    loss_perc_per_comp += [round(np.mean(loss_perc_per_comp),2)]
    split_stats = [lost_data.shape[1]-1,
                   X_train.shape[1]-13,
                   round(lost_data.shape[1]/X_train.shape[1]*100,2),
                   lost_data["hits_drop_per_comp"]["hits_dropped_per_substructure"],
                   total_hits,
                   round(lost_data["hits_drop_per_comp"]["hits_dropped_per_substructure"]/total_hits*100,2),
                   loss_perc_per_comp[-1]
                   ]
    names = ["features dropped","total features","% features dropped","hits dropped","total hits","% hits dropped","avg % of hits dropped (comp)"]

    while len(split_stats)!= lost_data.shape[0]:
        names.append("")
        split_stats.append("")
    lost_data["split_percentage_per_comp"] = loss_perc_per_comp
    lost_data["split_statistic"] = names
    lost_data[""] = split_stats
    lost_data.to_csv(r'./output/split_analysis.csv', index=True, sep=';')
    df = pd.DataFrame(index=[0])
    df["hits_dropped"] = lost_data["hits_drop_per_comp"]["hits_dropped_per_substructure"]
    df["features_dropped"] = lost_data.shape[1]-4
    for i in range(len(lost_columns)):
        df[f"{i}"]= lost_columns[i]


    df.to_csv("./output/split_history.csv",mode="a",sep=";",header=False,index=False)
    return X_train_slice,X_test_slice

