import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np

plot_kwargs = {"family": 'serif',
               "color": 'r',
               "weight": 'normal',
               "size": 16,
               "labelpad": 6
               }

hist_kwargs = {"bins" : (40, 40),
               "range" : [[-0.1, 0.11], [-0.1, 0.11]],
                "cmap" : plt.cm.rainbow
               }


def ax_limits():
    ax = [-0.1, 0.11, -0.1, 0.11]
    return ax


def limits():
    x = [-1, 1]
    y1 = [-0.99, 1.01]
    y2 = [-1.01, 0.99]
    y3 = [-0.98, 1.02]
    y4 = [-1.02, 0.98]
    return x, y1, y2, y3, y4

def plot_error(err_train, std_train, err_test, std_test, diff_error_train, err_avg_train, diff_error_test, err_avg_test):
    fig, axs = plt.subplots(1,4, figsize=(30,30))
    x, y = [0,0.1], [0,0.05]
    for i in range(4):
        axs[i].set_box_aspect(1)
    axs[0].scatter(err_train,std_train,s=9,c='green')
    axs[0].set_xlabel("error_train",
                      **plot_kwargs
                      )
    axs[0].set_ylabel("std_pred",
                      **plot_kwargs)
    axs[1].scatter(err_test,std_test,s=9,c='red')
    axs[1].set_xlabel("error_test",
                      **plot_kwargs
                      )
    axs[1].set_ylabel("std_pred",
                      **plot_kwargs)
    k = int(len(err_train)/2)
    j = int(len(err_test)/2)
    try:
        axs[2].scatter(diff_error_train[0:k], err_avg_train[0:k], s=9, c='green')
        axs[2].set_xlabel("diff error (train)",
                          **plot_kwargs
                          )
        axs[2].set_ylabel("average prediction error (train)",
                          **plot_kwargs)
        axs[3].scatter(diff_error_test[0:j], err_avg_test[0:j], s=9, c='red')
        axs[3].set_xlabel("diff error (test)",
                          **plot_kwargs
                          )
        axs[3].set_ylabel("average prediction error (test)",
                          **plot_kwargs)
    except TypeError:
        print("asymmetric data, no symmetry error")


    for i in range(4):
        axs[i-1].axis([0,0.01,0,0.01])
    for i in range(2):
        axs[i+2].plot(x,y,'g--')
    plt.savefig('./output/error')

def plot_training(y_train, y_pred_train, y_test, y_pred_test):
    from Dataset import num_targets
    fig, axs = plt.subplots(1, num_targets, figsize=(14, 14))
    plot_names_pred = ["m_seg_pred","sig_pred","eps/k_pred"]
    plot_names_true = ["m_seg", "sig", "eps/k"]

    for i in range(num_targets):
        axs[i].set_box_aspect(1)
        axs[i].scatter(y_train[:,i], y_pred_train[:,i], s=5, c='green')
        axs[i].scatter(y_test[:,i], y_pred_test[:,i], s=5, c='red')

        axs[i].set_xlabel(plot_names_true[i],
                          **plot_kwargs
                          )
        axs[i].set_ylabel(plot_names_pred[i],
                          **plot_kwargs)
        x, y1, y2, y3, y4 = limits()
        if num_targets == 1:
            axs[i].plot(x, y1, 'g--')
            axs[i].plot(x, y2, 'g--')
            axs[i].plot(x, x)
            axs[i].plot(x, y3, 'b--')
            axs[i].plot(x, y4, 'b--')
        elif num_targets==3:
            axs[i].plot([-1000,1000], [-1000,1000])
    if num_targets ==1:
        axs[0].axis= ax_limits()
    elif num_targets==3:
        axs[0].axis([1,10.5,1,10.5])
        axs[1].axis([2,5,2,5])
        axs[2].axis([100,400,100,400])
    #axs[1].hist2d(y_test,
    #              y_pred_test,
    #              **hist_kwargs
    #           )
    #axs[2].hist2d(y_train,
    #              y_pred_train,
    #              **hist_kwargs
    #           )
    plt.savefig('./output/output')
    plt.close()

def result_export(mean_train,std_train,y_test,error,diff_error, comp, name):
    mean_train = pd.DataFrame(mean_train)
    std_train = pd.DataFrame(std_train)
    y_test = pd.DataFrame(y_test)
    err_2 = error_threshold(error, 0.02)
    err_1 = error_threshold(error, 0.01)
    err_avg = list(np.reshape(np.add(error[0::2],error[1::2])/2,(int(len(error)/2),)))
    MAE = np.mean(error)
    MDE = np.mean(diff_error)
    DE_max = np.max(diff_error)
    err_stat = [err_2,err_1,MAE,MDE,DE_max]
    for i in range(len(error)-len(err_stat)):
        err_stat.append('')
    comp_half = comp[::2]
    for i in range(len(error)-len(diff_error)):
        err_avg.append('')
        diff_error.append('')
        comp_half.append('')
    error = pd.DataFrame(error)
    comp = pd.DataFrame(comp)
    data=pd.concat([mean_train,std_train,y_test, error,comp],axis=1)
    data.columns=["kij_pred_µ","kij_pred_std", "kij_test","abs_error","components"]
    data = data.sort_values(by=["kij_test"],ascending=False)
    data['error_stats'] = err_stat
    data['comp_single'] = comp_half
    data['kij-kji'] = diff_error
    data['kij-kji.pred_mean'] = err_avg
    data.to_csv(r'./output/'+name+'.csv', index=True, sep=';')
    return err_avg

def AARD(y_true,y_pred):
    from Dataset import num_targets
    AARD = []
    RMSE = []
    for k in range(num_targets):
        AARD_temp =[]
        RMSE_temp= []
        for i in range(len(y_true)):
            AARD_temp.append(100*abs(1-y_pred[i,k]/y_true[i,k]))
            RMSE_temp.append((y_pred[i,k]-y_true[i,k])**2)
        AARD.append(np.mean(AARD_temp).round(2))
        RMSE.append(((np.mean(RMSE_temp))**0.5).round(2))
    return AARD, RMSE

def result_export_nonsymmetric(mean_train,std_train,y_test,error, comp, name):
    from Dataset import num_targets
    from data_extraction import AARD
    AARD, RMSE = AARD(y_test,mean_train)
    mean_train = pd.DataFrame(mean_train)
    std_train = pd.DataFrame(std_train)
    y_test = pd.DataFrame(y_test)
    error=pd.DataFrame(error)
    correct_side_dev = 0
    print(error.shape[0])
    for i in range(error.shape[0]):
        if error.iloc[i,0]*error.iloc[i,1]<0 and error.iloc[i,0]*error.iloc[i,2]<0:
            correct_side_dev += 1
    print(correct_side_dev)
    print(error)
    err_stat = []
    if num_targets ==1:
        err_2 = error_threshold(error, 0.02)
        err_1 = error_threshold(error, 0.01)
        err_stat = [err_2, err_1]
    for i in range(num_targets):
        err_stat.append(np.mean(error.iloc[:,i]))
    err_stat.append(AARD)
    err_stat.append(RMSE)
    for i in range(len(error)-len(err_stat)):
        err_stat.append('')
    comp = pd.DataFrame(comp)
    data=pd.concat([mean_train,std_train,y_test, error,comp],axis=1)
    if num_targets==1:
        data.columns=["kij_pred_µ","kij_pred_std", "kij_test","abs_error","components"]
        data = data.sort_values(by=["abs_error"],ascending=False)
    elif num_targets==3:
        data.columns = ["seg_num_pred", "seg_dia_pred", "disp_energy_pred", "seg_num","seg_dia","disp_energy","err1","err2","err3","components"]
    data['error_stats'] = err_stat
    data.to_csv(r'./output/'+name+'.csv', index=True, sep=';')
    return AARD,RMSE


def error_threshold(error,limit):
    k=0
    for i in error:
        if i > limit:
            k+=1
    perc = (1-k/len(error))*100
    return perc

