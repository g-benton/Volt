import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd

import gpytorch
from torch.nn.functional import softplus
from voltron.kernels import BMKernel, VolatilityKernel
from voltron.models import BMGP, VoltronGP
import argparse
from torch.distributions import Beta
from scipy.special import betainc
from Trainers import *

def main(args):
    full_data = pd.read_pickle("../../spdr-data/" + args.SPDR + ".pkl")
    tckrs = full_data.symbol.unique()
    
    for tckr in tckrs:
        data = full_data[full_data["symbol"] == tckr]

        ts = torch.linspace(0, data.shape[0]/252., data.shape[0])
        # train_x = ts[:ntrain]
        # test_x = ts[ntrain:(ntrain+ntest)]

        y = torch.FloatTensor(data['close_price'].to_numpy())
        log_returns = torch.log(y[1:]) - torch.log(y[:-1])
        dt = ts[1] - ts[0]
        
        eval_times = list(range(100, ts.shape[0], 100)) #+ [ts.shape[0]]
        prob_of_increases = []

        for i, time in enumerate(eval_times):
            print("now running time: ", time)
            with gpytorch.settings.max_cholesky_size(2000):
                data_model, data_lh = get_and_fit_basic_model(ts[:time], y[:time], 
                                                    cov=args.kernel, mean=args.mean)
                end_ind = -1 if i + 1 >= len(eval_times) else eval_times[i+1]
                paths = predict_basic_prices(ts[time:end_ind], data_model,
                                            data_lh).detach()
                # now we predict the probability of increase at time i + 1
                prob_of_increase = (paths[..., -1] > y[time]).sum() / paths.shape[-2]
                print("prob of stock increase: ", prob_of_increase.detach())

            prob_of_increases.append(prob_of_increase.detach())
            
        torch.save(obj=prob_of_increases, f="./outputs/" + args.kernel + "_" + tckr + ".pt")
        print(tckr, "Done")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--SPDR",
        type=str,
        default="XLE",
    )    
    parser.add_argument(
        "--kernel",
        type=str,
        default="matern",
    ) 
    parser.add_argument(
        "--mean",
        type=str,
        default="loglinear",
    )  
    args = parser.parse_args()

    main(args)