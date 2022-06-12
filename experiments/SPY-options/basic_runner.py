import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gpytorch
import os
# import robin_stocks.robinhood as r
import pickle5 as pickle
import pandas as pd
import argparse

import sys
sys.path.append("../")
from voltron.likelihoods import VolatilityGaussianLikelihood
from voltron.models import SingleTaskVariationalGP as SingleTaskCopulaProcessModel
from voltron.kernels import BMKernel, VolatilityKernel
from voltron.models import BMGP, VoltronGP
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from voltron.option_utils import GetTradingDays, GetTrainingData, Pricer, FindLastTradingDays
from voltron.train_utils import TrainBasicModel

def main(args):
    years = [yr for yr in range(2006, 2018)]
    logger = []
    full_logger = []
    SPY = pd.read_csv("./data/SPY_prices.csv")
    SPY['Date'] = pd.to_datetime(SPY['Date'])
    ntrain = 252
    
    nvol = 100
    npx = 100
    
    for year in years:
        options = pd.read_csv("./data/SPY_" + str(year) + ".csv")
        options.expiration = pd.to_datetime(options.expiration)
        options.quotedate = pd.to_datetime(options.quotedate)
        qday = options.quotedate.unique()[0]
        quote_price = SPY[SPY['Date']==qday].Close.item()
        options = options[(options.quotedate == qday) & (options.type=='call')]
        edays = options.expiration.sort_values().unique()
        testdays = (edays - qday)/np.timedelta64(1, "D")
        edays = edays[(testdays > 100) & (testdays < 365)]
        lastdays = FindLastTradingDays(SPY, edays)
        ntests = np.array([GetTradingDays(SPY, qday, pd.Timestamp(ld)) for ld in lastdays])
        fulltest = ntests[-1]

        train_y = torch.FloatTensor(GetTrainingData(SPY, qday, ntrain).to_numpy())
        test_y = torch.FloatTensor(GetTrainingData(SPY, 
                                                   pd.Timestamp(lastdays[-1]),
                                                   fulltest).to_numpy())
        full_x = torch.arange(ntrain+fulltest).type(torch.FloatTensor)
        full_x = full_x/252.
        train_x = full_x[:ntrain]
        test_x = full_x[ntrain:]

        dmod, dlh = TrainBasicModel(train_x, train_y, train_iters=500, model_type=args.model,
                                   mean_func=args.mean_func)
    
        ## figure out how to price options sanely ##

        nvol = 100
        npx = 100
        px_samples = torch.zeros(npx*nvol, len(edays))
        px_paths = torch.zeros(npx*nvol, fulltest)
        dmod.eval();

        for vidx in range(nvol):
            px_pred = dlh(dmod(test_x)).sample(torch.Size((npx,))).exp()
            px_paths[vidx*npx:(vidx*npx + npx), :] = px_pred.detach()
            px_samples[vidx*npx:(vidx*npx+npx), :] = px_pred[:, ntests-1].detach()
    
    
        
        option_output = Pricer(px_samples, options, edays, test_y[ntests-1],
                               quote_price)
        option_output.to_pickle("./output/" + args.model + "_options" + str(year) + ".pkl")
        print(str(year), "Done")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mean_func",
        type=str,
        default="loglinear",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="matern",
    )
    
    args = parser.parse_args()

    main(args)