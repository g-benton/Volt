import pandas as pd
import yfinance as yf
import datetime
import numpy as np


def make_ticker_list(file_name):
    tickers = open(file_name, 'r')
    tickers = [i.strip() for i in list(tickers)]
    return tickers

def make_price_files(tickers, start, end, fpath, printing):
    for i in tickers:
        history = yf.download(tickers=i, 
                              start=start, 
                              end=end, 
                              progress=False,
                             )
        history.to_csv(fpath + str(i) + '.csv')
        if printing:
            print(str(i))
        

def DataGetter(history = 500, fpath="../data/", printing=False, end_date=None,
               ticker_file="test_tickers.txt"):
    if end_date is None:
        end_date = datetime.date.today()
    else:
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
        
    start_date = end_date - datetime.timedelta(history)
    end_date = str(end_date)

    tickers = make_ticker_list(fpath + ticker_file)
    make_price_files(tickers, start_date, end_date, fpath, printing)
    
def GetStockHistory(ticker, end_date=str(datetime.date.today()), history=500):
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    data = yf.download(tickers=ticker, period='10y', progress=False)
    end_idx = np.where(data.index == pd.to_datetime(end_date))[0][0]
    
    return data.iloc[end_idx-history:end_idx]
    
    
    
