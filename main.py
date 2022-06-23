import pickle
import torch
import torchvision
from copy import copy
from pathlib import Path
from torch.utils.data import DataLoader
from Dataset import *
from Model import *
from data_extraction import *
import random

def main(rand_seed,lr,wd):

    torch.manual_seed(2)
    X_train, X_test, y_train, y_test, comp_train, comp_test, feature_loss,scale_param, y_test_pre,y_train_pre = load_raw_data(rand_seed)
    X_test[:,11:] = X_test[:,11:].round(0)
    X_train[:, 11:] = X_train[:, 11:].round(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())
    build_dir = Path("build")
    if not build_dir.exists():
        build_dir.mkdir()

    model_pLV = loss_Model().to(device)
    model_pLV.load_state_dict(torch.load('./loss_model/state_dict_pLV.pt'))
    model_pLV.eval()
    scaler_pLV = torch.tensor(pd.read_csv("./loss_model/scaler_parameters_pLV.csv",sep=";",header=None).to_numpy()).to(device)


    model_rho = loss_Model().to(device)
    model_rho.load_state_dict(torch.load('./loss_model/state_dict_rho.pt'))
    model_rho.eval()
    scaler_rho = torch.tensor(pd.read_csv("./loss_model/scaler_parameters_rho.csv",sep=";",header=None).to_numpy()).to(device)
    scaler_loss = torch.cat((scaler_rho,scaler_pLV))



    train_dataset = Dataset_input(X_train,y_train,comp_train)
    test_dataset = Dataset_input(X_test,y_test,comp_test)



    #print(torch.from_numpy(np.array([1.2,1.0,0.1])).to(device))
    loader_kwargs = {
        "batch_size": 64,
        "drop_last": False,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "shuffle":False
    }
    train_loader = DataLoader(dataset=train_dataset,
                              **loader_kwargs
                              )
    test_loader = DataLoader(dataset=test_dataset,
                             **loader_kwargs
                             )



    model_path = build_dir / "nn.torch"
    model = Model().to(device)
    model.apply(weights_init_uniform_rule)
    model.to(device)

    if model_path.exists():
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        loss_fn = torch.nn.GaussianNLLLoss(eps=1e-12)
        optimizer = torch.optim.Adam(model.parameters(),
                                     weight_decay=wd,  #6e-6
                                     lr=lr  #0.0048
                                     )
        from Dataset import input_length, n_samples, symm_mode, bin_mode, num_targets, std_eval
        epochs = 10000
        start_var = 10000
        loss_min = 0.5
        iter_best = "none"
        loss_history = {"train": [], "validation": []}

        for t in range(epochs):
            print(
                f"\nEpoch {t + 1}/{epochs}"
                "\n-------------------------------"
                f"\ntrain (seed:{seed}):"
            )
            if t < start_var:
                train(train_loader, model, loss_fn, optimizer, scale_param,model_pLV,model_rho,scaler_loss, loss_history["train"],fit_variance=False)
            else:
                train(train_loader, model, loss_fn, optimizer, scale_param,model_pLV,model_rho,scaler_loss,loss_history["train"],fit_variance=True)

            print("validate:")
            if t < start_var:
                loss = validate(test_loader, model, loss_fn,model_pLV, model_rho, scaler_loss, scale_param,loss_history["validation"], fit_variance=False)
            else:
                loss = validate(test_loader, model, loss_fn, model_pLV, model_rho, scaler_loss, scale_param,loss_history["validation"], fit_variance=True)
            print(f"loss: {loss:.5f}")
            if loss < loss_min:
                loss_min = loss.item()
                iter_best = t
            if loss < 0.002: #0.0013
                break
            elif loss_min >= 2 and t > 600:
                break

        visualize_training(
            loss_history["train"],
            loss_history["validation"],
            build_dir / "loss.png",
        )
        with open(build_dir / "loss.pickle", "wb") as f:
            pickle.dump(loss_history, f)
        # torch.save(model.state_dict(), model_path)
        torch.save(model,
                   './build/model_gauss')

        print("\nTraining done!")
        print(loss_min)
        print(iter_best)

    def calc_pred(data_loader, model,model_rho,model_pLV):
        model.eval()
        mean = []
        std = []
        y = []
        for Xi, yi in data_loader:
            if symm_mode == 1:
                Xi = Xi.to(device)
                yi = yi.to(device)
            elif symm_mode == 2:
                Xi = Xi.to(device).reshape(-1, input_length)
                yi = yi.to(device).reshape(-1, 1)

            pred = model(Xi.to(device))  ### shape = (num_targets*2 für std_eval)
            pred_res = pred.cpu().detach().numpy()
            loss_pc_rho, loss_pc_pLV = loss_PCSAFT(yi,pred,model_pLV,model_rho,scaler_loss,scale_param)
            num_targets = yi.size(1)
            num_outputs = pred.size(1)

            if num_targets == num_outputs:
                mean.extend(pred_res.tolist())
                y.extend(yi.cpu().detach().numpy().tolist())
            else:
                mean.extend(pred_res[:, 0:(num_outputs - 1)].tolist())
                y.extend(yi.cpu().detach().numpy().tolist())
                std.extend(pred_res[:, -num_outputs:])
            print(f"PC-SAFT loss (Rho): {loss_pc_rho:.2f} ; PC-SAFT loss (pLV): {loss_pc_pLV:.2f}")
        mean = retransform(mean)
        y = retransform(y)
        abs_error = []
        diff_error = []
        for i in range(len(y)):
            abs_error.append(mean[i] - y[i]) #oder abs für korrekten MAE
        if symm_mode == 2:
            for l in range(int(len(mean) / 2)):
                diff_error.append(abs(mean[2 * l] - mean[2 * l + 1]))
        return mean, std, y, abs_error, diff_error,loss_pc_pLV.cpu().detach().numpy(),loss_pc_rho.cpu().detach().numpy()


    pred_test_mean, pred_test_std, y_test, abs_error_test, diff_error_test,loss_pc_pLV_test,loss_pc_rho_test = calc_pred(test_loader, model, model_rho,model_pLV)
    pred_train_mean, pred_train_std, y_train, abs_error_train, diff_error_train,loss_pc_pLV_train,loss_pc_rho_train = calc_pred(train_loader, model,model_rho,model_pLV)



    #print(pred_train_mean)
    # print(pred_train_std)
    #print(y_train)

    #print(pred_test_mean)
    # print(pred_test_std)
    #print(y_test)

    if (len(pred_test_mean) + len(pred_train_mean)) != n_samples:
        err_avg_train = result_export(pred_train_mean,
                                      pred_train_std,
                                      y_train,
                                      abs_error_train,
                                      diff_error_train,
                                      comp_train,
                                      'output_train')
        err_avg_test = result_export(pred_test_mean,
                                     pred_test_std,
                                     y_test,
                                     abs_error_test,
                                     diff_error_test,
                                     comp_test,
                                     'output_test')
    else:
        AARD_train,RMSE_train = result_export_nonsymmetric(pred_train_mean,
                                                   pred_train_std,
                                                   y_train,
                                                   abs_error_train,
                                                   comp_train,
                                                   'output_train')
        AARD_test,RMSE_test = result_export_nonsymmetric(pred_test_mean,
                                                  pred_test_std,
                                                  y_test,
                                                  abs_error_test,
                                                  comp_test,
                                                  'output_test')

    plot_training(y_train,
                  pred_train_mean,
                  y_test,
                  pred_test_mean,
                  )
    if bin_mode == 2:
        plot_error(abs_error_train,
                   pred_train_std,
                   abs_error_test,
                   pred_test_std,
                   diff_error_train,
                   err_avg_train,
                   diff_error_test,
                   err_avg_test
                   )
    return loss_min,iter_best,AARD_test,RMSE_test,feature_loss,loss_pc_pLV_test,loss_pc_rho_test


loss_threshold = 0.1
steps = 1
steps_seed = 1000
seed_start = 1
seed_save=[]
loss_min_save=[]
iter_best_save=[]
AARD_save=[]
RMSE_save=[]
loss_pc_rho_save = []
loss_pc_pLV_save = []
lr_save=[]
wd_save=[]
feature_loss_save=[]
seed_list = [seed_start+steps for steps in range(steps_seed)]
lr_base= 2.8e-5#0.00033
wd_base = 3.75e-5#2.74e-5

for i in range(steps):
    lr_temp = lr_base
    wd_temp = wd_base
    #lr_temp = round(lr_base + random.uniform(-2e-5,5e-5),7)
    #wd_temp = round(wd_base + random.uniform(-2e-5,4.5e-5),7)
    print(lr_temp,wd_temp)
    k=0
    for seed in seed_list:
        k+=1
        if (lr_temp,wd_temp) not in list(zip(lr_save,wd_save)) or seed_list[0] != seed:
        #if 0.0 == 0.0:
            print(f"------------------------------\nstart calculation for seed:{seed}")
            print(f"lr used:{lr_temp}; wd used: {wd_temp}")
            loss_min_i, iter_best_i, AARD_test_i, RMSE_test_i, feature_loss_i,loss_pc_pLV_test,loss_pc_rho_test = main(seed, lr_temp, wd_temp)
            print(
                f"----------------------------\n{i * len(seed_list) + k}/{len(seed_list) * steps} combinations checked\n"
                f"Approximated calculation time left: {round((len(seed_list) * steps - (i * len(seed_list) + k)) * 0.9, 0)} min")

            if loss_min_i < loss_threshold:
                seed_save.append(seed)
                iter_best_save.append(iter_best_i)
                loss_min_save.append(loss_min_i)
                lr_save.append(lr_temp)
                wd_save.append(wd_temp)
                AARD_save.append(AARD_test_i)
                RMSE_save.append(RMSE_test_i)
                loss_pc_rho_save.append(loss_pc_rho_test)
                loss_pc_pLV_save.append(loss_pc_pLV_test)
                feature_loss_save.append(feature_loss_i)

            #try:
            #    print(f"best_loss: {min(loss_min_save):.6f}\n"
            #          f"seeds found: {len(loss_min_save)}\n------------------------------------")
            #except:
            #    continue
        else:
            print(f"lr:{lr_temp}; wd: {wd_temp} already used!")
            continue
        try:
            print(f"best_loss: {round(min(loss_min_save), 6)}\n"
                  f"seeds found: {len(loss_min_save)}\n------------------------------------")
        except:
            print("no loss_min found yet!")
            pass

print(loss_min_save,seed_save)
export = pd.DataFrame(data=np.array([seed_save,loss_min_save,iter_best_save,lr_save,wd_save,AARD_save,RMSE_save,feature_loss_save,loss_pc_rho_save,loss_pc_pLV_save],dtype=object),
                      index=["seed","loss","iter_best","lr","wd","AARD","RMSE","len_X","AARD_rho","RMSE_plv"]).transpose()
if steps != 1 or len(seed_list) !=1:
    export.to_csv('./output/optim.csv',sep=";")