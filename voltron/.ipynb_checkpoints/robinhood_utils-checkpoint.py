import robin_stocks.robinhood as r
import os
import pandas as pd
from dotenv import load_dotenv

def GetStockData(symbols, interval='day', span='5year'):
    """
    just a wrapper for robin-stocks calls
    """
    load_dotenv()
    username = os.getenv("robinhood_username")
    password = os.getenv("robinhood_password")
    r.login(username, password);
    
    data = pd.DataFrame(r.stocks.get_stock_historicals(symbols, interval, span))
    data['date'] = pd.to_datetime(data['begins_at'], format='%Y-%m-%d').dt.date
    
    ohlc = ['open_price', 'close_price', 'high_price', 'low_price']
    data[ohlc] = data[ohlc].astype("float")
    
    return data[['date', 'symbol', 'open_price', 'close_price', 
                'high_price', 'low_price']]