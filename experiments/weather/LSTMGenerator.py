import numpy as np
import torch
import pandas as pd
import gpytorch
import argparse
import datetime
import warnings
import os
from voltron.data import make_ticker_list, GetStockHistory
import sys
sys.path.append("../calibration")
from LSTMUtils import SequenceDataset, LSTM, TrainLSTM, LSTMRollouts, NLL
from torch.utils.data import DataLoader
import pickle as pkl

def main(args):
    
    stn_names, stn_lonlat, full_data = pkl.load(open("./wind_data.p", 'rb'))
    
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    
    ntest = args.forecast_horizon
    ntrain = args.ntrain
    seq_len = args.seq_length
    n_test_times = args.n_test_times
    ntime = full_data[0].shape[0]
    
    test_idxs = torch.arange(ntrain, ntime-ntest, 
                         int((ntime-ntest-ntrain)/n_test_times))
    
    stn_idxs = list(stn_names.keys())
    
    for stn in stn_idxs:
        savepath = "./saved-outputs/stn" + str(stn) + "/"
        stn_data = full_data[stn]
        stn_data[stn_data == -99.0] = 0.
        if stn_data.mean() != 0:
            
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            for last_day in test_idxs:
                try:
                    raw_y = stn_data[last_day-ntrain:last_day] + 1
                    raw_y = torch.FloatTensor(raw_y).log()
                    train_y = (raw_y - raw_y.mean())/raw_y.std()
                    if use_cuda:
                        train_y = train_y.cuda()

                    ## make trainloader ##
                    dset = SequenceDataset(train_y, seq_len)
                    trainloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True)

                    model = LSTM(2, seq_len, 128, 1)
                    if use_cuda:
                        model = model.cuda()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    TrainLSTM(trainloader, model, NLL, optimizer, 
                              epochs=args.train_epochs,
                              printing=False, use_cuda=use_cuda)

                    rollouts = LSTMRollouts(model, args.nsample, ntest,
                                            dset, use_cuda).cpu()
                    rollouts = rollouts * raw_y.std() + raw_y.mean()
                    torch.save(rollouts, savepath + "lstm_" + str(last_day.item()) + ".pt")
                    print("stn ", stn, " idx ", last_day.item())
                except:
                    print("### BROKEN stn", stn, " idx", last_day, " ###")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--seq_length",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=400,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
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
        default=200,
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
        default=False,
    )
    args = parser.parse_args()

    main(args)