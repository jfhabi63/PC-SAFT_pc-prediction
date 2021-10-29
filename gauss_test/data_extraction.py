import pandas as pd
import matplotlib.pyplot as plt

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


def plot_training(y_train, y_pred_train, y_test, y_pred_test):
    fig, axs = plt.subplots(1, 3, figsize=(14, 14))
    for i in range(3):
        axs[i].set_box_aspect(1)
    axs[0].scatter(y_train, y_pred_train, s=5, c='green')
    axs[0].scatter(y_test, y_pred_test, s=5, c='red')
    axs[0].axis(ax_limits())
    axs[0].set_xlabel("kij_test",
                      **plot_kwargs
                      )
    axs[0].set_ylabel("kij_pred",
                      **plot_kwargs)
    x, y1, y2, y3, y4 = limits()
    axs[0].plot(x, y1, 'g--')
    axs[0].plot(x, y2, 'g--')
    axs[0].plot(x, x)
    axs[0].plot(x, y3, 'b--')
    axs[0].plot(x, y4, 'b--')
    axs[1].hist2d(y_test,
                  y_pred_test,
                  **hist_kwargs
               )
    axs[2].hist2d(y_train,
                  y_pred_train,
                  **hist_kwargs
               )
    plt.savefig('./output/output')

def result_export(mean_train,std_train,y_test,error, name):
    mean_train = pd.DataFrame(mean_train)
    std_train = pd.DataFrame(std_train)
    y_test = pd.DataFrame(y_test)
    error = pd.DataFrame(error)

    data=pd.concat([mean_train,std_train,y_test, error],axis=1)
    data.to_csv(r'./output/'+name+'.csv', index=True, header=["kij_pred_µ","kij_pred_std", "kij_test","abs_error"], sep=';')
