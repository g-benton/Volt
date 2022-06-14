import numpy as np
import torch
import pandas as pd
import gpytorch
import argparse
import datetime
import warnings
import os
from voltron.data import make_ticker_list, GetStockHistory
from LSTMUtils import SequenceDataset, LSTM, TrainLSTM, LSTMRollouts, NLL
from torch.utils.data import DataLoader

def main(args):
    
    data_path = "../../voltron/data/"
    ticker_file = args.ticker_fname + ".txt"
    tckr_list = make_ticker_list(data_path + ticker_file)
#     tckr_list = ['TSLA']
    
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    
    ntest = args.forecast_horizon
    ntrain = args.ntrain
    seq_len = args.seq_length
    
    if args.end_date.lower() == "none":
        end_date = datetime.date.today()
    else:
        end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d")
    

    for tckr in tckr_list:
        try:
            data = GetStockHistory(tckr, history= ntrain + args.lookback)
            end_idxs = torch.arange(args.ntrain, data.shape[0],
                       int((data.shape[0]-args.ntrain)/args.ntimes))

            savepath = "./saved-outputs/" + tckr + "/"
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            for last_day in end_idxs:
                date = str(data.index[last_day.item()].date())
                raw_y = data.Close[last_day.item()-ntrain:last_day.item()].to_numpy()
                raw_y = torch.FloatTensor(raw_y).log()
                train_y = (raw_y - raw_y.mean())/raw_y.std()

                ## make trainloader ##
                dset = SequenceDataset(train_y, seq_len)
                trainloader = DataLoader(dset, batch_size=args.batch_size, shuffle=True)

                model = LSTM(2, seq_len, 128, 1)
                if use_cuda:
                    model = model.cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                TrainLSTM(trainloader, model, NLL, optimizer, epochs=args.train_epochs,
                         printing=True, use_cuda=use_cuda)

                rollouts = LSTMRollouts(model, args.nsample, ntest,
                                        dset, use_cuda).cpu()
                rollouts = rollouts * raw_y.std() + raw_y.mean()
                torch.save(rollouts, savepath + "lstm_" + date + ".pt")

                del model
        except:
            print("FAILED ", tckr)
            
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
        default=20,
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=25,
    )
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