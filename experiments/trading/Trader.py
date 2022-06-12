import numpy as np
import torch
import pandas as pd
import gpytorch
import argparse

def ValueFunction(pred_samples, curr_px):
    """
    pred_samples = (num samples) x (test times) matrix of forecast paths
    curr_px = last observed price
    """
    snr = (pred_samples.mean(0) - curr_px)/pred_samples.std(0)
    
    if snr > 0.2:
        return 1.
    else:
        return 0.



def main(args):
    data_path = "../../voltron/data/"
    ticker_file = "test_tickers.txt"
    tckr_list = make_ticker_list(data_path + ticker_file)
    
    if args.end_date
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kernel",
        type=str,
        default="volt",
    )    
    parser.add_argument(
        "--mean",
        type=str,
        default="ewma",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
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
        default=400,
    )
    parser.add_argument(
        "--end_date",
        default="none",
    )
    args = parser.parse_args()

    main(args)