import numpy as np
import torch
import pandas as pd
import gpytorch
import argparse
import datetime
import warnings

from voltron.data import make_ticker_list, GetStockHistory
from GenerateMultiMeanPreds import GenerateStockPredictions, GenerateBasicPredictions, GenerateOneDayPredictions
from gpytorch.utils.warnings import NumericalWarning
warnings.simplefilter("ignore", NumericalWarning)
warnings.simplefilter("ignore", UserWarning)

def main(args):
    tckr = args.ticker
    ## download data ##
    if args.end_date.lower() == "none":
        end_date = datetime.date.today()
    else:
        end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
        
    dat = GetStockHistory(tckr, history=args.ntrain + args.lookback,
                          end_date=str(end_date))
    
    ## pick a day to generate forecasts ##
    end_idxs = torch.arange(args.ntrain, dat.shape[0],
                   int((dat.shape[0]-args.ntrain)/args.ntimes))
    last_day = end_idxs[args.test_idx]
    date = str(dat.index[last_day.item()].date())
    
    print(date, tckr)
    
    train_y = torch.FloatTensor(dat.Close[last_day.item()-args.ntrain:last_day.item()].to_numpy())
    
    GenerateOneDayPredictions(tckr, train_y, date, 
                              forecast_horizon=args.forecast_horizon,
                            train_iters=args.train_iters, 
                             nsample=args.nsample,
                            ntrain=400, save=args.save, mean=args.mean)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntimes",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--test_idx",
        type=int,
        default=0,
    )
    
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default='ADBE',
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=400,
    )
    parser.add_argument(
        '--kernel',
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
        "--printing",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--end_date",
        default="none",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    main(args)