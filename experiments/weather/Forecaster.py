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
from torch.utils.data import DataLoader
from voltron.models import LSTM, BasicGP, Volt
import pickle as pkl

def main(args):
    
    ################
    ## Data Setup ##
    ################
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
    stn_data = full_data[stn]
    stn_data[stn_data == -99.0] = 0.
    
    train_x = torch.arange(ntrain).float()/365
    test_x = torch.arange(ntrain, ntrain + ntest).float()/365

    if use_cuda:
        train_x, test_x = train_x.cuda(), test_x.cuda()
    
    ####################
    ## setup filename ##
    ####################
    savepath = "./saved-outputs/stn" + str(stn) + "/"
    
    if args.model.lower() == 'lstm':
        modelname = "lstm_"
    else:
        if args.model.lower() == 'gp':
            modelname = "gp_" + args.kernel + "_"
        elif args.model.lower() == 'volt':
            modelname = "volt_"
        
        if args.mean.lower() == 'constant':
            modelname += 'constant' + "_"
        elif args.mean.lower() in ['ewma', 'dewma', 'tewma']:
            modelname += args.mean + args.k + "_"

    
    ###############
    ## Main Loop ##
    ###############
    
    if stn_data.mean() != 0:
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        for last_day in test_idxs:
#             try:
            raw_y = stn_data[last_day-ntrain:last_day] + 1
            train_y = torch.FloatTensor(raw_y).log()
            if use_cuda:
                train_y = train_y.cuda()


            if args.model.lower() == 'lstm':
                model = LSTM(train_x, train_y, 10, 128, 1)
                model.Train(args.train_iters)
            elif args.model.lower() == 'gp':
                model = BasicGP(train_x, train_y, kernel=args.kernel,
                                mean=args.mean, k=args.k)
                model.Train(args.train_iters)
            elif args.model.lower() == 'volt':
                model = Volt(train_x, train_y, mean=args.mean, k=args.k)
                model.Train(gpcv_iters=args.train_iters, 
                            vol_mod_iters=args.train_iters,
                            data_mod_iters=args.train_iters)
            else:
                print("ERROR: Model not found")


            samples = model.Forecast(test_x).squeeze()
            torch.save(samples, savepath + modelname + str(last_day.item()) + ".pt")
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
        default=20,
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--model",
        type=str,
        default='volt',
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
        "--train_iters",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=400,
    )
    
    args = parser.parse_args()

    main(args)