import numpy as np
import torch
import pandas as pd
import gpytorch
import argparse
import datetime
import warnings
import copy
import os
from voltron.data import make_ticker_list, GetStockHistory
import sys
sys.path.append("../calibration")
from LSTMUtils import SequenceDataset, LSTM, TrainLSTM, LSTMRollouts, NLL
from torch.utils.data import DataLoader
from voltron.train_utils import LearnGPCV, TrainVolModel, TrainVoltMagpieModel, TrainBasicModel 
from voltron.rollout_utils import Rollouts
from BasicWind import BasicWindRollouts
import pickle as pkl

def main(args):
    
    stn_names, stn_lonlat, full_data = pkl.load(open("./wind_data.p", 'rb'))
    
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    
    stn = args.stn_idx
    ntest = args.forecast_horizon
    ntrain = args.ntrain
    n_test_times = args.n_test_times
    ntime = full_data[0].shape[0]
    
    test_idxs = torch.arange(ntrain, ntime-ntest, 
                         int((ntime-ntest-ntrain)/n_test_times))
    
    stn_idxs = list(stn_names.keys())
    if args.kernel == 'volt':
        train_x = torch.arange(ntrain-1).float()/365
    else:
        train_x = torch.arange(ntrain).float()/365
    test_x = torch.arange(ntrain, ntrain + ntest).float()/365

    if use_cuda:
        train_x, test_x = train_x.cuda(), test_x.cuda()
    
    savepath = "./saved-outputs/stn" + str(stn) + "/"
    stn_data = full_data[stn]
    stn_data[stn_data == -99.0] = 0.
    if stn_data.mean() != 0:
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        for last_day in test_idxs:
#             try:
            raw_y = stn_data[last_day-ntrain:last_day] + 1
            train_y = torch.FloatTensor(raw_y)
            if use_cuda:
                train_y = train_y.cuda()

            if args.kernel == 'volt':
                with gpytorch.settings.max_cholesky_size(2000):
                    vol = LearnGPCV(train_x, train_y, train_iters=200,
                                        printing=False)
                    vmod, vlh = TrainVolModel(train_x, vol, 
                                              train_iters=500, printing=False)

                if args.mean == 'constant':
                    voltron, lh = TrainVoltMagpieModel(train_x, train_y[1:], 
                           vmod, vlh, vol,
                           printing=False, 
                           train_iters=200, mean_func="constant")
                    vmod.eval();
                    voltron.eval();
                    voltron.vol_model.eval();
                    theta = 0.01
#                         for theta in [0., 0.01, 0.025, 0.05, 0.1]:

                    temp_model = copy.deepcopy(voltron)
                    with torch.no_grad():
                        save_samples = Rollouts(train_x, train_y, test_x, temp_model, 
                                        nsample=args.nsample, theta=theta)
                    torch.save(save_samples, savepath + args.kernel + "_theta" + str(theta) +\
                               "_" + str(last_day.item()) + ".pt")

                    del temp_model

                else:
                    for k in [400]:
                        voltron, lh = TrainVoltMagpieModel(train_x, train_y[1:], 
                               vmod, vlh, vol,
                               printing=False, 
                               train_iters=0, mean_func="ewma", k=k)
                        vmod.eval();
                        voltron.eval();
                        voltron.vol_model.eval();
                        for theta in [0.01]:
                            temp_model = copy.deepcopy(voltron)
                            with torch.no_grad():
                                save_samples = Rollouts(train_x, train_y, 
                                                        test_x, temp_model, 
                                                nsample=args.nsample, theta=theta)
                            torch.save(save_samples, savepath + args.kernel + "_ema" + str(k) +\
                                       "_theta" + str(theta) +\
                                       "_" + str(last_day.item()) + ".pt")
                            del temp_model
                    del voltron, vmod, vol, vlh
            else:
                k=200
                rollouts = BasicWindRollouts(train_x, train_y, test_x,
                                             train_iters=args.train_epochs,
                                            kernel_name=args.kernel,
                                            mean_name=args.mean, k=k,
                                            nsample=200)
                
                torch.save(rollouts, savepath + args.kernel + "_" +\
                           args.mean + str(k) + "_" + str(last_day.item()) + ".pt")

            print("stn ", stn, " idx ", last_day.item())

            torch.cuda.empty_cache()
#             except:
#                 print("### BROKEN stn", stn, " idx", last_day, " ###")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stn_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--mean",
        type=str,
        default='constant',
    )
    parser.add_argument(
        "--n_test_times",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="matern",
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--printing",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    main(args)