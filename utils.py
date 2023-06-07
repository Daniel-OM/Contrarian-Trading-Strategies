
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from plotly.subplots import make_subplots


def getData(asset:str, tf:str='1h', start:str=None, end:str=None):

    '''
    Parameters
    ----------
    asset: str
        Instrument to download data for.
    tf: str
        Timeframe of the data. Default is 1h.
    start: str
        Date from which to download.
    end: str
        Date till which to download. It will only be used if the start 
        date is specified.
    '''

    ticker = yf.Ticker(asset)
    
    # Check if the start was specified
    if start != None:

        # Check if start is in the correct format
        if isinstance(start, dt.date) or isinstance(start, dt.datetime):
            start = start.strftime('%Y-%m-%d')

        # Check if the end was specified
        if end != None:
            # Check if end is in the correct format
            if isinstance(end, dt.date) or isinstance(end, dt.datetime):
                end = end.strftime('%Y-%m-%d')
            data = ticker.history(interval=tf, start=start, end=end)

        # If no end was specified
        else:
            data = ticker.history(interval=tf, start=start, 
                                  end=dt.datetime.today().strftime('%Y-%m-%d'))

    # If no start was specified
    else:
        data = ticker.history('max',tf)
        if data.empty:
            end = dt.datetime.today()
            start = end - dt.timedelta(days=220*3)
            data = ticker.history(interval=tf, 
                                start=start.strftime('%Y-%m-%d'), 
                                end=end.strftime('%Y-%m-%d'))
    
    complete = {
        'data': data,
        'info': ticker.info,
    }
          
    return complete 

def indices():

    url = 'https://finance.yahoo.com/world-indices'
    ua = "Gozilla/5.0" # Mozilla
    r = requests.get(url, headers={'User-Agent': ua})

    indices = pd.read_html(r.text)[0]
    indices_symbol = indices['Symbol'].tolist()

    return indices

def performance(data:pd.DataFrame, open_price:str='Open', buy_column:str='Buy', 
                sell_column:str='Short', long_result_col:str='LongResult', 
                short_result_col:str='ShortResult', total_result_col:str='Total'
                ) -> pd.DataFrame:
    
    sell = []
    cover = []
    long_result = []
    short_result = []
    
    # Check long trades result
    for i,idx in enumerate(data.index):
        d = data.iloc[i] # Current candle
        next = data.iloc[i+1:] # Next candles

        # If there was a buy order
        if d[buy_column] == 1:
            # Check for the next sell order
            for a in next.index:
                new_d = next.loc[a]
                if new_d[buy_column] == 1 or new_d[sell_column] == -1:
                    # Store result
                    long_result.append(new_d[open_price] - d[open_price])
                    sell.append(1)
                    short_result.append(0)
                    cover.append(0)
                    break

        # If there was a sell order
        elif d[sell_column] == -1:    
            for a in next.index:
                new_d = next.loc[a]
                if new_d[buy_column] == 1 or new_d[sell_column] == -1:
                    short_result.append(d[open_price] - new_d[open_price])
                    cover.append(-1)
                    long_result.append(0)
                    sell.append(0)
                    break

        else:
            sell.append(0)
            cover.append(0)
            long_result.append(0)
            short_result.append(0)
            
    data['Sell'] = sell[len(sell) - len(data):] if len(sell) > len(data) \
                    else [0]*(len(data) - len(sell)) + sell
    data['Cover'] = cover[len(cover) - len(data):] if len(cover) > len(data) \
                    else [0]*(len(data) - len(cover)) + cover
            
    data[long_result_col] = long_result[len(long_result) - len(data):] if len(long_result) > len(data) \
                    else [0]*(len(data) - len(long_result)) + long_result
    data[short_result_col] = short_result[len(short_result) - len(data):] if len(short_result) > len(data) \
                    else [0]*(len(data) - len(short_result)) + short_result
        
    # Aggregating the long & short results into one column
    data[total_result_col] = data[long_result_col] + data[short_result_col]  
    data['Cum'+total_result_col] = data[total_result_col].cumsum()
    
    # Profit factor    
    total_net_profits = data[total_result_col][data[total_result_col] > 0]
    total_net_losses = data[total_result_col][data[total_result_col] < 0]
    total_net_losses = abs(total_net_losses)
    profit_factor = round(np.sum(total_net_profits) / np.sum(total_net_losses), 2)

    # Hit ratio    
    hit_ratio = len(total_net_profits) / (len(total_net_losses) + len(total_net_profits))
    hit_ratio = hit_ratio
    
    # Risk reward ratio
    average_gain = total_net_profits.mean()
    average_loss = total_net_losses.mean()
    realized_risk_reward = average_gain / average_loss

    # Number of trades
    trades = len(total_net_losses) + len(total_net_profits)
        
    print('Number of Trades  = ', trades)    
    print('Profit factor     = ', profit_factor) 
    print('Hit Ratio         = ', hit_ratio * 100)
    print('Realized RR       = ', round(realized_risk_reward, 3))
    print('Expectancy        = ', hit_ratio*average_gain - (1-hit_ratio)*average_loss)
   
    return data

def ma(data:pd.DataFrame, lookback:int, close:str='Close', name:str='MA'): 

    if not isinstance(data, pd.DataFrame):
        raise ValueError('data not in DataFrame format.')
    if close not in data.columns:
        raise ValueError(f'{close} not in dataframe columns.')
    
    data[name] = data[close].rolling(lookback).mean()
    
    return data

def signalChart(df:pd.DataFrame, asset:str='', indicators:list=[]):

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Candlestick(x=df.index,open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'], 
                                name='Price'), row=1, col=1)
    
    # Plot indicators
    for ind in indicators:
        fig.add_trace(go.Scatter(x=df.index,y=df[ind],name=ind))

    # Long trades
    fig.add_trace(go.Scatter(x=df.index[df['Buy'] > 0], y=df['Open'][df['Buy'] > 0], 
                            name='Buy', marker_color='green', marker_symbol='triangle-right', 
                            marker_size=15, mode='markers'), 
                row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index[df['Sell'] > 0], y=df['Open'][df['Sell'] > 0], 
                            name='Sell', marker_color='green', marker_symbol='triangle-left', 
                            marker_size=15, mode='markers'), 
                row=1, col=1)

    # Short trades
    fig.add_trace(go.Scatter(x=df.index[df['Short'] < 0], y=df['Open'][df['Short'] < 0], 
                            name='Short', marker_color='red', marker_symbol='triangle-right', 
                            marker_size=15, mode='markers'), 
                row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index[df['Cover'] < 0], y=df['Open'][df['Cover'] < 0], 
                            name='Cover', marker_color='red', marker_symbol='triangle-left', 
                            marker_size=15, mode='markers'), 
                row=1, col=1)

    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_layout(title=f"Price {asset}", autosize=False,
                        xaxis_rangeslider_visible=False,
                        width=1000,
                        height=700)

    fig.show()

    return fig

if __name__ == '__main__':

    data = getData('QQQ')['data']