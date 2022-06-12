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
sns.set_style("whitegrid")
sns.set_palette("bright")
sns.set(font_scale=2.0)


import sys
sys.path.append("../")
from voltron.likelihoods import VolatilityGaussianLikelihood
from voltron.models import SingleTaskVariationalGP as SingleTaskCopulaProcessModel
from voltron.kernels import BMKernel, VolatilityKernel
from voltron.models import BMGP, VoltronGP
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from voltron.option_utils import GetTradingDays, GetTrainingData, Pricer, FindLastTradingDays
from voltron.train_utils import LearnGPCV, TrainVolModel, TrainDataModel

def main():
    years = [yr for yr in range(2006, 2018)]
    logger = []
    full_logger = []
    SPY = pd.read_csv("./data/SPY_prices.csv")
    SPY['Date'] = pd.to_datetime(SPY['Date'])
    ntrain = 375
    
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
        dt = train_x[1]-train_x[0]
        test_x = full_x[ntrain:]

        ## learn vol with GPCV ##
        vol = LearnGPCV(train_x, train_y, train_iters=750)/(dt**0.5)
        
        ## train vol GP ## 
        vmod, vlh = TrainVolModel(train_x, vol, train_iters=750)
        
        ## train data gp ##
        dmod, dlh = TrainDataModel(train_x, train_y, vmod, vlh, vol,
                printing=False, train_iters=750)
    
        ## figure out how to price options sanely ##

        px_samples = torch.zeros(npx*nvol, len(edays))
        px_paths = torch.zeros(npx*nvol, fulltest)
        vol_paths = torch.zeros(nvol, fulltest)
        dmod.vol_model.eval();
        dmod.eval();

        for vidx in range(nvol):
        #     print(vidx)
            vol_pred = dmod.vol_model(test_x).sample().exp()
            vol_paths[vidx, :] = vol_pred.detach()

            px_pred = dmod.GeneratePrediction(test_x, vol_pred, npx).exp()
            px_paths[vidx*npx:(vidx*npx + npx), :] = px_pred.detach().T
            px_samples[vidx*npx:(vidx*npx+npx), :] = px_pred[ntests-1].detach().T
    
        
        option_output = Pricer(px_samples, options, edays, test_y[ntests-1],
                               quote_price)
        option_output.to_pickle("./output/options" + str(year) + ".pkl")
        print(str(year), "Done")
    
    
if __name__ == "__main__":
    main()
    
