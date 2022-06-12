import numpy as np
import torch
import pandas as pd
import gpytorch
import argparse
import datetime
import warnings

from voltron.data import make_ticker_list, GetStockHistory
from GenerateMultiMeanPreds import GenerateStockPredictions, GenerateBasicPredictions, GenerateGPCVPredictions
from gpytorch.utils.warnings import NumericalWarning
warnings.simplefilter("ignore", NumericalWarning)

def main(args):
    
    
    data_path = "../../voltron/data/"
    ticker_file = args.ticker_fname + ".txt"
    tckr_list = make_ticker_list(data_path + ticker_file)
    
    if args.end_date.lower() == "none":
        end_date = datetime.date.today()
    else:
        end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    for tckr in tckr_list:
        try:
            data = GetStockHistory(tckr, history=args.ntrain + args.lookback,
                                  end_date=str(end_date))
        except:
            print(tckr, "FAILED")
            data = None
        if data is not None:
            if args.kernel.lower() == 'volt':
                GenerateStockPredictions(tckr, data, forecast_horizon=args.forecast_horizon,
                                        train_iters=args.train_iters, 
                                         nsample=args.nsample,
                                        ntrain=400, save=args.save, ntimes=args.ntimes,
                                        vol_kernel=args.vol_kernel.lower())
            elif args.kernel.lower() == 'gpcv':
                GenerateGPCVPredictions(tckr, data, forecast_horizon=args.forecast_horizon,
                                        train_iters=args.train_iters, 
                                         nsample=args.nsample,
                                        ntrain=400, ntimes=args.ntimes)

            else:
                GenerateBasicPredictions(tckr, data, forecast_horizon=args.forecast_horizon,
                                         kernel_name=args.kernel, mean_name=args.mean, k=args.k,
                                            train_iters=args.train_iters, 
                                             nsample=args.nsample,
                                            ntrain=args.ntrain, save=args.save)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--ticker_fname",
        type=str,
        default='nasdaq100',
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
        '--vol_kernel',
        type=str,
        default="bm",
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
        default='2022-04-08',
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