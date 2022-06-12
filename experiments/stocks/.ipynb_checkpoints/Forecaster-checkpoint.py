import numpy as np
import torch
import pandas as pd
import gpytorch
import argparse
import datetime
import warnings

from voltron.data import make_ticker_list, GetStockHistory
import sys
sys.path.append("../trading/")
from GenerateMultiMeanPreds import GenerateStockPredictions, GenerateBasicPredictions
from gpytorch.utils.warnings import NumericalWarning
warnings.simplefilter("ignore", NumericalWarning)

def main(args):
    
    if args.end_date is None:
        end_date = datetime.date.today()
    else:
        end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")

    data = GetStockHistory(args.ticker, history=args.ntrain + args.lookback)
    
    ntest = args.forecast_horizon
    ntrain = args.ntrain
    n_test_times = args.n_test_times
    ntime = data.shape[0]
    
    test_idxs = torch.arange(ntrain, ntime-ntest, 
                         int((ntime-ntest-ntrain)/n_test_times))

    train_x = torch.arange(ntrain) * dt
    test_x = torch.arange(ntest) * dt + train_x[-1] + dt
    
    if torch.cuda.is_available():
        train_x = train_x.cuda()
        test_x = test_x.cuda()
    
    ####################
    ## setup filename ##
    ####################
    savepath = "./saved-outputs/" + args.ticker + "/"
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    if args.model.lower() == 'lstm':
        modelname = "lstm"
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
    
    for last_day in test_idxs:
        date = str(data.index[last_day.item()].date())
        train_y = data.Close[last_day.item()-ntrain:last_day.item()].to_numpy()
        train_y = torch.FloatTensor(train_y).to(train_x.device)
        
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
                torch.save(samples, savepath + modelname + date + ".pt")
                torch.cuda.empty_cache()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ticker",
        type=str,
        default='F',
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--n_test_times",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=100,
    )
    parser.add_argument(
        '--kernel',
        type=str,
        default="matern",
    )
    parser.add_argument(
        '--model',
        type=str,
        default="volt",
    )
    parser.add_argument(
        '--mean',
        type=str,
        default="ewma",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--end_date",
        default=None,
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    main(args)