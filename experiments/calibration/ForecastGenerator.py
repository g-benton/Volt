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
    
    
    data_path = "../../voltron/data/"
    ticker_file = args.ticker_fname + ".txt"
    tckr_list = make_ticker_list(data_path + ticker_file)
    
    if args.end_date.lower() == "none":
        end_date = datetime.date.today()
    else:
        end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
    

    for tckr in tckr_list:
        try:
            data = GetStockHistory(tckr, history=args.ntrain + args.lookback)
            if args.kernel.lower() == 'volt':
                GenerateStockPredictions(tckr, data, forecast_horizon=args.forecast_horizon,
                                        train_iters=args.train_iters, 
                                         nsample=args.nsample, mean_name=args.mean,
                                        ntrain=args.ntrain, save=args.save)
            else:
                GenerateBasicPredictions(tckr, data, forecast_horizon=args.forecast_horizon,
                                         kernel_name=args.kernel, mean_name=args.mean,
                                         k=args.k, train_iters=args.train_iters, 
                                             nsample=args.nsample, ntimes=args.ntimes,
                                                    ntrain=args.ntrain, save=args.save)

        except:
            print("FAILED ", tckr)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ticker_fname",
        type=str,
        default='test_tickers',
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--ntimes",
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