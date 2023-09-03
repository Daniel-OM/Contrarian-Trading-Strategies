
import copy
import datetime as dt
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import yfinance as yf
from backtest import (AssetConfig, BackTest, BtConfig, Commissions,
                      StrategyConfig)
#from google_sheets.google_sheets import GoogleSheets
from degiro import DeGiro, IntervalType, Product, ResolutionType
from indicators import Indicators
from signals import Signals

config = BtConfig('2000-01-01', dt.date.today().strftime('%Y-%m-%d'), 
                    capital=2000.0, monthly_add=200,  # (dt.date.today() - dt.timedelta(days=250)).strftime('%Y-%m-%d')
                    use_sl=True, use_tp=True, time_limit=None, min_size=1000, 
                    max_size=10000000, commission=Commissions(), max_trades=1000, 
                    filter_ticker=False, filter_strat=False, reset_orders=True,
                    continue_onecandle=False, offset_aware=False)


tickers = {'SP500': {'yfinance':'500.PA', 'degiro':'LU1681048804'}, 
            'NASDAQ': {'yfinance':'ANX.PA', 'degiro':'LU1681038243'}, 
            'STOXX': {'yfinance':'C50.PA'}, 
            'MSCI World': {'yfinance':'CW8.PA'}}

# Prepare all the available strategies
signals = Signals(backtest=True, side=Signals.Side.LONG, errors=False)
strategies = {s: StrategyConfig(name=s, assets={
                'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1') \
            for s in dir(signals) if '_' not in s and 'get' not in s and not s[0].isupper() and \
                callable(getattr(signals, s))}

strategies = {
    'atrExt': StrategyConfig(name='atrExt', assets={
                'SP500': AssetConfig(name='SPY', risk=0.0075, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1', filter='MA_100'),
 'dailyPB': StrategyConfig(name='dailyPB', assets={
                'SP500': AssetConfig(name='SPY', risk=0.008, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1', filter='MA_100'),
 'detrended': StrategyConfig(name='detrended', assets={
                'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1', filter='MA_100'), # 2
 'envelopes': StrategyConfig(name='envelopes', assets={
                'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
 'fibTiming': StrategyConfig(name='fibTiming', assets={
                'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
 'rsiExtremeDuration': StrategyConfig(name='rsiExtremeDuration', assets={
                'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'), # 2
 'chandeMomentum': StrategyConfig(name='chandeMomentum', assets={
                'SP500': AssetConfig(name='SPY', risk=0.04, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1'),
 'macdTrend': StrategyConfig(name='macdTrend', assets={
                'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
 'paraSar': StrategyConfig(name='paraSar', assets={
                'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1'), # 2
 'pullbackBounce': StrategyConfig(name='pullbackBounce', assets={
                'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
 'rsiAtr': StrategyConfig(name='rsiAtr', assets={
                'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
 'stochExtreme': StrategyConfig(name='stochExtreme', assets={
                'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
 'trendContinuation': StrategyConfig(name='trendContinuation', assets={
                'SP500': AssetConfig(name='SPY', risk=0.002, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1', filter='MA_100'),
 'trendInten': StrategyConfig(name='trendInten', assets={
                'SP500': AssetConfig(name='SPY', risk=0.04, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'), # 2
 'turtlesBreakout': StrategyConfig(name='turtlesBreakout', assets={
                'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1', filter='MA_100'),
 'volatPB': StrategyConfig(name='volatPB', assets={
                'SP500': AssetConfig(name='SPY', risk=0.005, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1', filter='MA_100')}

strategies = {
    'detrended': StrategyConfig(name='detrended', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1', filter='MA_100'), # 2
    'envelopes': StrategyConfig(name='envelopes', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'fibTiming': StrategyConfig(name='fibTiming', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'rsiExtremeDuration': StrategyConfig(name='rsiExtremeDuration', assets={
        'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'), # 2
    'chandeMomentum': StrategyConfig(name='chandeMomentum', assets={
        'SP500': AssetConfig(name='SPY', risk=0.04, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1'),
    'macdTrend': StrategyConfig(name='macdTrend', assets={
        'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'rsiAtr': StrategyConfig(name='rsiAtr', assets={
        'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'stochExtreme': StrategyConfig(name='stochExtreme', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'trendContinuation': StrategyConfig(name='trendContinuation', assets={
        'SP500': AssetConfig(name='SPY', risk=0.005, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1', filter='MA_100'),
    'trendInten': StrategyConfig(name='trendInten', assets={
        'SP500': AssetConfig(name='SPY', risk=0.04, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'), # 2
    'turtlesBreakout': StrategyConfig(name='turtlesBreakout', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1', filter='MA_100'),
    'dailyPB': StrategyConfig(name='dailyPB', assets={
        'SP500': AssetConfig(name='SPY', risk=0.015, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1', filter='MA_100'),
    'volatPB': StrategyConfig(name='volatPB', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=3, timeframe='D1'),
    'pullbackBounce': StrategyConfig(name='pullbackBounce', assets={
        'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1'),
    'atrExt': StrategyConfig(name='atrExt', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=10.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=False, time_limit=10, timeframe='D1', filter='MA_100'),
    'kamaTrend': StrategyConfig(name='kamaTrend', assets={
        'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=10.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=10, timeframe='D1'),
    'rsiNeutrality': StrategyConfig(name='rsiNeutrality', assets={
        'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=10, timeframe='D1'),
    'paraSar': StrategyConfig(name='paraSar', assets={
        'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=5.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'momentum': StrategyConfig(name='momentum', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=5.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'adxMomentum': StrategyConfig(name='adxMomentum', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=100, timeframe='D1'),
    'weeklyDip': StrategyConfig(name='weeklyDip', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'stochDip': StrategyConfig(name='stochDip', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
}


#Connect to the data provider
broker = 'degiro'
if broker == 'degiro':
    dg = DeGiro('OneMade','Onemade3680')

# Prepare data needed for backtest
indicators = Indicators(errors=False)
total_data = []
data = {}

portfolio = True
apply_filter = True
if portfolio:
    if len(data) <= 0:
        for strat in strategies:

            if strat not in dir(signals):
                print(f'{strat} not between the defined signals.')
                continue
            signal = getattr(signals, strat)

            for t,c in strategies[strat].assets.items():

                if t not in data:
                    if broker == 'yfinance':
                        temp = yf.Ticker(tickers[t][broker]).history(period='5y',interval='1d')
                    elif broker == 'degiro':
                        products = dg.searchProducts(tickers[t][broker])
                        temp = dg.getCandles(Product(products[0]).id, resolution=ResolutionType.D1, 
                                             interval=IntervalType.Max)
                else:
                    temp = data[t].copy()

                temp.loc[:,'distATR'] = indicators.atr(n=20, method='s', dataname='ATR', new_df=temp)['ATR']
                temp.loc[:,'SLdist'] = temp['distATR'] * c.sl
                temp.loc[:,'Ticker'] = [t]*len(temp)
                temp.loc[:,'Date'] = pd.to_datetime(temp.index)
                if 'DateTime' not in temp.columns and 'Date' in temp.columns:
                    temp.rename(columns={'Date':'DateTime'}, inplace=True)
                
                if 'Volume' not in temp.columns:
                    temp['Volume'] = [0]*len(temp)
                    
                temp = signal(df=temp, strat_name=strat)
                if apply_filter and strategies[strat].filter != None and 'MA' in strategies[strat].filter:
                    period = int(strategies[strat].filter.split('_')[-1])
                    temp['filter'] = temp['Close'].rolling(period).mean()
                    temp[temp.columns[-2]] = np.where(temp['Close'] > temp['filter'], temp[temp.columns[-2]], 0)

                if t in data:
                    for c in temp:
                        if c not in data[t]:
                            data[t][c] = temp[c]
                else:
                    data[t] = temp.copy()

    # Prepare data
    df = pd.concat([data[t] for t in data], ignore_index=True)
    df.sort_values('DateTime', inplace=True)
    df.loc[:,'DateTime'] = pd.to_datetime(df['DateTime'], unit='s')
    if df['DateTime'].iloc[0].tzinfo != None:
        df['DateTime'] = df['DateTime'].dt.tz_convert(None)
    df['Close'] = df['Close'].fillna(method='ffill')
    df['Spread'] = df['Spread'].fillna(method='ffill')
    df['Open'] = np.where(df['Open'] != df['Open'], df['Close'], df['Open'])
    df['High'] = np.where(df['High'] != df['High'], df['Close'], df['High'])
    df['Low'] = np.where(df['Low'] != df['Low'], df['Close'], df['Low'])

    # Backtest
    bt = BackTest(strategies=strategies, 
                config=config)

    trades = bt.backtest(df=df)

    bt.btPlot(balance=True, log=True)
    stats = bt.stats(column='StratName')

else:
    trades = {}
    for strat in strategies:

        data = {}
        if strat not in dir(signals):
            print(f'{strat} not between the defined signals.')
            continue
        signal = getattr(signals, strat)

        for t,c in strategies[strat].assets.items():

            if t not in data:
                if broker == 'yfinance':
                    temp = yf.Ticker(tickers[t][broker]).history(period='5y',interval='1d')
                elif broker == 'degiro':
                    products = dg.searchProducts(tickers[t][broker])
                    temp = dg.getCandles(Product(products[0]).id, interval=IntervalType.Max)
            else:
                temp = data[t].copy()

            temp.loc[:,'distATR'] = indicators.atr(n=20, method='s', dataname='ATR', new_df=temp)['ATR']
            temp.loc[:,'SLdist'] = temp['distATR'] * c.sl
            temp.loc[:,'Ticker'] = [t]*len(temp)
            temp.loc[:,'Date'] = pd.to_datetime(temp.index)
            if 'DateTime' not in temp.columns and 'Date' in temp.columns:
                temp.rename(columns={'Date':'DateTime'}, inplace=True)
            
            if 'Volume' not in temp.columns:
                temp['Volume'] = [0]*len(temp)

            temp = signal(df=temp, strat_name=strat)
            if apply_filter and strategies[strat].filter != None and 'MA' in strategies[strat].filter:
                period = int(strategies[strat].filter.split('_')[-1])
                temp['filter'] = temp['Close'].rolling(period).mean()
                temp[temp.columns[-2]] = np.where(temp['Close'] > temp['filter'], temp[temp.columns[-2]], 0)

            data[t] = temp.copy()

        # Prepare data
        df = pd.concat([data[t] for t in data], ignore_index=True)
        df.sort_values('DateTime', inplace=True)
        df.loc[:,'DateTime'] = pd.to_datetime(df['DateTime'], unit='s')
        if df['DateTime'].iloc[0].tzinfo != None:
            df['DateTime'] = df['DateTime'].dt.tz_convert(None)
        df['Close'] = df['Close'].fillna(method='ffill')
        df['Spread'] = df['Spread'].fillna(method='ffill')
        df['Open'] = np.where(df['Open'] != df['Open'], df['Close'], df['Open'])
        df['High'] = np.where(df['High'] != df['High'], df['Close'], df['High'])
        df['Low'] = np.where(df['Low'] != df['Low'], df['Close'], df['Low'])

        # Backtest
        bt = BackTest(strategies={strat: strategies[strat]}, 
                    config=config)

        trades[strat] = bt.backtest(df=df)
        print(strat, '---------------------------------------------------------------------')
        if trades[strat].empty:
            print('No trades')
        else:
            bt.btPlot(balance=True, log=True)
            stats = bt.stats(column='StratName')
