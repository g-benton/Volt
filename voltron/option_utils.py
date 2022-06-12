import numpy as np
import torch
import pandas as pd


def GetTrainingData(SPY, date, N):
    idx = SPY[SPY["Date"] == date].index.item()
    return SPY['Close'].iloc[(idx-N):idx]

def GetTrueValue(SPY, date, strike):
    close_px = SPY['Close'][SPY["Date"] == date].item()
    return np.maximum(close_px-strike, 0)

def GetTradingDays(SPY, start, stop):
    start_idx = SPY[SPY["Date"] == start].index.item()
    stop_idx = SPY[SPY["Date"] == stop].index.item()
    return stop_idx-start_idx

def FindLastTradingDays(SPY, dates):
    last_days = []
    for date in dates:
        last_days.append(np.max(np.where(SPY.Date < date)[0]))
        
    return np.array(SPY.Date[last_days])

def Pricer(mc_pxs, options, edays, true_pxs, quote_price):
    logger = []
    for eday_idx, eday in enumerate(edays):
        eday = pd.Timestamp(eday)
        year = pd.DatetimeIndex([eday])[0].year
        opts = options[options.expiration==pd.Timestamp(eday)]
        for idx, row in opts.iterrows():
            K = row.strike
            bid = row.bid
            ask = row.ask
            valuation = np.mean(np.maximum(mc_pxs[:, eday_idx].numpy() - K, 0))
            rtn = np.maximum(true_pxs[eday_idx] - K, 0)
            pct = ECDF(mc_pxs[:, eday_idx], true_pxs[eday_idx])
            logger.append([eday, K, bid, ask, valuation, rtn.item(),
                           true_pxs[eday_idx].item(), quote_price, year, pct])
            
    df = pd.DataFrame(logger)
    df.columns = ['Expiry', "Strike", "Bid", "Ask", "Voltron", "Return",
                 "ExpClose", "QuoteClose", "Year", "Sample_Percentile"]
    return df


def ECDF(sample_pxs, true_px):
    smp = sample_pxs.log().sort()[0]
    log_px = true_px.log()
    
    return (torch.sum(smp < log_px)/smp.shape[0]).item()