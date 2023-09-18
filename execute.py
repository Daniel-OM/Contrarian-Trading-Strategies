
import copy
import datetime as dt
import os
import random
import time

import numpy as np
import pandas as pd
#from google_sheets.google_sheets import GoogleSheets
from backtest import StrategyConfig
from config import (apply_filter, broker, end_time, execute, start_time,
                    strategies, tickers, trades_url, open_trades_name)
from degiro import (DataType, DeGiro, IntervalType, Order, Product,
                    ResolutionType)
from indicators import Indicators
from signals import Signals


class Trade:

    def __init__(self, data:pd.DataFrame=None, signal:str=None, strategy:StrategyConfig=None,
                 balance:float=None) -> None:

        '''
        Generates the trade object for the backtest.

        Parameters
        ----------
        candle: dict
            Dictionary with the current candle data.
        signal: str
            Direction for the trade. It can be 'long' or 'short'.
        strategy: str
            Name of the strategy used to enter the trade.
        entry: float
            Entry price.
        balance: float
            Balance when entering the trade.
        '''
        
        self.entrytime =  dt.datetime.today().strftime('%Y-%m-%d %H:%M')
        self.exittime = None
        self.exit = None
        self.returns = None
        self.balance = balance
        self.signal = signal
        self.result = None

        if strategy != None:
            strategy = copy.deepcopy(strategy)
            self.strategy = strategy

        else:
            self.asset = None

        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
            candle = data[-1]
            self.candle = candle
            self.datetime = candle['DateTime']
            self.ticker = candle['Ticker']
            self.entry = candle['Open']
        else:
            self.ticker = None
            self.entry = None

        if (isinstance(data, pd.DataFrame) or isinstance(data, list)) and \
            strategy != None:

            asset = copy.deepcopy(strategy.assets[candle['Ticker']])
            self.asset = asset
            self.order = asset.order_type
            self.entry = self.calculateEntry(candle, data[-2])
            self.risk = asset.risk
            if self.ticker == None:
                self.ticker = asset.name
            sldist = asset.sl * candle['distATR']
            tpdist = asset.tp * candle['distATR']
            self.sldist = sldist
            self.sl = self.entry - sldist if signal == 'long' else self.entry + sldist
            self.tp = self.entry + tpdist if signal == 'long' else self.entry - tpdist

        self.size = self.calculateSize()

    def calculateEntry(self, candle:dict=None, prev_candle:dict=None) -> float:

        if candle == None or prev_candle == None:
            raise ValueError('Candle data is needed for the Entry to be calculated')
        
        entry = candle['Open']
        if self.signal == 'long':
            if self.order == 'stop':
                if 'High' in prev_candle and \
                    prev_candle['High'] > candle['Open']:
                    entry = prev_candle['High']
                else:
                    self.order = 'market'
            elif self.order == 'limit':
                if 'Low' in prev_candle and \
                    prev_candle['Low'] < candle['Open']:
                    entry = prev_candle['Low']
                else:
                    self.order = 'market'
        elif self.signal == 'short':
            if self.order == 'stop':
                if 'Low' in prev_candle and \
                    prev_candle['Low'] < candle['Open']:
                    entry = prev_candle['Low']
                else:
                    self.order = 'market'
            elif self.order == 'limit':
                if 'High' in prev_candle and \
                    prev_candle['High'] > candle['Open']:
                    entry = prev_candle['High']
                else:
                    self.order = 'market'

        return entry

    def calculateSize(self, risk:float=None, balance:float=None, 
                      sldist:float=None) -> float:

        '''
        Calculates the size of the trade.

        Returns
        -------
        size: float
            Size of the trade.
        '''
        
        if risk != None:
            self.risk = risk
        if balance != None:
            self.balance = balance
        if sldist != None:
            self.sldist = sldist
        if self.risk == None or self.balance == None or self.sldist == None or self.asset == None \
            or self.asset == None:
            print(self.risk, self.balance, self.sldist, self.asset)
            raise ValueError('There is not enough data to calculate the \
                             trade size. An asset should be specified for the trade.')

        self.size = int(self.risk * self.balance / self.sldist)
        if self.size > self.asset.max_size:
            self.size = self.asset.max_size
        elif self.size < self.asset.min_size:
            self.size = self.asset.min_size
        
        if self.size * self.entry > self.balance:
            self.size = int(self.balance/self.entry)

        if self.balance <= 0:
            self.size = 0.0

        return self.size

    def exitTrade(self, exit:float) -> None:

        '''
        Calculates the Exit variables.

        Parameters
        ----------
        exit: float
            Exit price.
        '''

        self.exittime = dt.datetime.today().strftime('%Y-%m-%d %H:%M')
        self.exit = exit
        self.returns = exit - self.entry if self.signal == 'long' else self.entry - exit
        self.result = self.returns * self.size
    
    def to_dict(self):

        '''
        Generates a dictionary with the trade data.

        Returns
        -------
        object: dict
            Contains the data.
        '''

        # self.exitTrade()
        self.strategy = self.strategy.to_dict()
        self.asset = self.asset.to_dict()

        return self.__dict__

class TradesFiles:

    def __init__(self, path:str='execution', verbose:bool=False) -> None:
        self.path = path
        self.verbose = verbose

    def _generateTrade(self) -> dict:

        strategy = list(strategies.values())[random.randint(0,len(strategies)-1)]
        ticker = list(strategy.assets.keys()) \
                [random.randint(0,len(strategy.assets) - 1)]
        
        open_p = random.randint(100000, 110000)/10000
        data = [{
            'DateTime': (dt.datetime.today() - dt.timedelta(days=random.randint(1,400))),
            'Open':random.randint(100000, 110000)/10000,
            'High': open_p * abs(1+random.gauss(0,0.02)),
            'Low': open_p * -abs(1+random.gauss(0,0.02)),
            'Close': open_p * (1+random.gauss(0,0.02)),
            'Volume': random.randint(1,10000),
            'Spread': 0,
            'distATR': random.randint(10, 100)/10000,
            'Ticker': ticker,
        }]
        open_p *= (1+random.gauss(0,0.02))
        data.append({
            'DateTime': (data[0]['DateTime'] + dt.timedelta(days=1)),
            'Open': open_p,
            'High': open_p * abs(1+random.gauss(0,0.02)),
            'Low': open_p * -abs(1+random.gauss(0,0.02)),
            'Close': open_p * (1+random.gauss(0,0.02)),
            'Volume': random.randint(1,10000),
            'Spread': 0,
            'distATR': random.randint(10, 100)/10000,
            'Ticker': ticker,
        })

        trade = Trade(
            data= pd.DataFrame(data),
            signal= 'long' if random.randint(0,1) else 'short',
            strategy= strategy,
            balance= random.randint(1000, 5000)
        )
        trade.exitTrade(exit=trade.entry * (1 + random.randint(-10, 10)/100000))
        
        return trade.to_dict()

    def _generateTrades(self, n:int=20) -> pd.DataFrame:

        return pd.DataFrame([self._generateTrade() for i in range(n)])

    def openTradesQty(self, file:str='open_trades.csv') -> int:

        data = self.getOpenTrades(file=file)

        return len(data)
    
    def getOpenTrades(self, file:str='open_trades.csv') -> pd.DataFrame:

        path = os.path.join(self.path, file)
        try:
            if '.csv' in file:
                trades = pd.read_csv(path)
            elif '.xlsx' in file:
                trades = pd.read_excel(path)
        except Exception as e:
            if e.__str__() == 'No columns to parse from file' and self.verbose:
                print('File is empty')
                trades = pd.DataFrame()

        return trades
    
    def addOpenTrades(self, data:pd.DataFrame, file:str='open_trades.csv', mode:str='a') -> None:

        path = os.path.join(self.path, file)
        exists = os.path.exists(path)
        trades_qty = self.openTradesQty(file=file)
        if '.csv' in file:
            data.to_csv(path, mode=mode, index=False, header=True \
                        if not exists or mode=='w' or trades_qty<=0 else False)
        elif '.xlsx' in file:
            with pd.ExcelWriter(path, mode=mode) as writer:
                data.to_excel(writer, index=False, header=True \
                            if not exists or mode=='w' or trades_qty<=0 else False)

    def deleteOpenTrade(self, data:pd.DataFrame, file:str='open_trades.csv') -> None:

        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        if isinstance(data, pd.Series):
            data = [dict(data)]
        if isinstance(data, dict):
            data = [data]

        old_data = self.getOpenTrades(file=file)
        # old_dict = {
        #     f"{d['entrytime']}-{d['ticker']}-{d['entry']}-{d['strategy']}": d for d in old_data.to_dict('records'),
        # }
        new_data = []
        for d in old_data.to_dict('records'):
            if d not in data:
                new_data.append(d)
        
        self.addOpenTrades(data=pd.DataFrame(new_data), file=file, mode='w')
    

    

def postOrder(trade:Trade=None, product_id:str=None, side:str=None, 
              order:str=None, entry:float=None, sl_dist:float=None,
              size:float=None) -> None:

    if trade != None:
        product_id = trade.asset.id
        side = trade.signal
        order = trade.order
        entry = trade.entry
        sl_dist = trade.sldist
        size = trade.size
    elif product_id == None or side == None or order == None or \
        entry == None or sl_dist == None or size == None:
        raise ValueError('Input a trade or some trade data.')
    
    product = dg.searchProducts(product_id)[0]
    side = 'BUY' if side == 'long' else 'SELL'
    
    limit = None
    if execute:

        if order == 'stoplimit':
            limit = entry + sl_dist/2 if side == 'BUY' \
                    else entry - sl_dist/2
            dg.tradeOrder(product, int(size), side, Order.Type.STOPLIMIT, 
                        Order.Time.GTC, limit=limit, stop_loss=entry)
            
        elif order == 'stop':
            dg.tradeOrder(product, int(size), side, Order.Type.STOPLOSS, 
                        Order.Time.GTC, stop_loss=entry)
            
        elif order == 'limit':
            dg.tradeOrder(product, int(size), side, Order.Type.LIMIT,
                        Order.Time.GTC, limit=entry)

        elif order == 'market':
            entry = dg.getQuote(product)['data']['lastPrice']
            limit = entry + sl_dist/2 if side == 'BUY' \
                    else entry - sl_dist/2
            dg.tradeOrder(product, int(size), side, Order.Type.STOPLIMIT, 
                        Order.Time.GTC, limit=limit, stop_loss=entry)
            
        else:
                dg.tradeOrder(product, int(size), side, Order.Type.MARKET, 
                            Order.Time.GTC)
                
    executions.append({'date':dt.datetime.today().strftime('%Y-%m-%d %H:%M'),
                       'product':product, 'size':int(size), 'side':side, 
                       'order':order, 'limit':limit, 'entry':entry})


def enterOrders(trades:dict) -> None:

    for symbol in trades:

        df = pd.DataFrame(trades[symbol])
        initial_size = df['size'].sum()
        df['size'] = np.where(df['signal'] == 'short', -df['size'], df['size'])

        total_size = df['size'].sum()
        if total_size > 0:
            side = 'long'
        elif total_size < 0:
            side = 'short'
        else:
            side = None

        if side != None:

            df = df[df['signal'] == side]
            df['size'] = np.where(df['signal'] == 'short', -df['size'], df['size'])
            for order in df.groupby('order'):

                size = order[1]['size'].sum()/initial_size * total_size 
                for entry in order['entry'].groupby('entry'):

                    # Check if there is enough balance 
                    balance = getEquity()
                    size = size if size*entry[0] <= balance else balance/entry[0]

                    # Prepare trade
                    trade = entry[1]['trade'].iloc[-1]
                    postOrder(product_id=symbol, side=side[0], order=order[0], 
                              entry=entry, sl_dist=trade.sldist, size=size)
                    
                    time.sleep(random.randint(40, 90))
        
def getEquity() -> float:

    return float(dg.getData(DataType.CASHFUNDS)[0][3:])

def getPortfolio() -> list:

    return dg.getData(DataType.PORTFOLIO)

def getQuote(product_id:str=None, product:Product=None):

    if product_id != None:
        product =  dg.searchProducts(product_id)[0]

    quote = dg.getQuote(product)['data']
    quote['tradingStartTime'] = dt.datetime.strptime(quote['tradingStartTime'], '%H:%M:%S').time()
    quote['tradingEndTime'] = dt.datetime.strptime(quote['tradingEndTime'], '%H:%M:%S').time()
    
    return quote

def time_interval(start:dt.time, end:dt.time, seconds:bool=True):

    reverse = False
    if start > end:
        start, end = end, start
        reverse = True

    delta = (end.hour - start.hour)*60 + end.minute - start.minute + (end.second - start.second)/60.0
    if reverse:
        delta = 24*60 - delta

    return delta * 60 if seconds else delta


def getData(strategies:dict, get_portfolio:bool=True):

    data = {}
    portfolio = {}
    for strat in strategies:

        # Prepare signal if exists
        if strat not in dir(signals):
            print(f'{strat} not between the defined signals.')
            continue
        signal = getattr(signals, strat)

        for s,c in strategies[strat].assets.items():
            
            strategies[strat].assets[s].id = tickers[s][broker]
            t = tickers[s][broker]
            
            # Get data
            if t not in data:
                product = dg.searchProducts(t)[0]

                # Check if market is open for the asset
                quote = getQuote(product=product)
                if dt.datetime.today().time() < quote['tradingStartTime'] or quote['tradingEndTime'] < dt.datetime.today().time():
                    data[t] = pd.DataFrame()
                    continue
                temp = dg.getCandles(Product(product).id, resolution=ResolutionType.H1, 
                                    interval=IntervalType.Y3)
            else:
                temp = data[t].copy()

            # Add columns
            temp['distATR'] = indicators.atr(n=20, method='s', dataname='ATR', new_df=temp)['ATR']
            temp['SLdist'] = temp['distATR'] * c.sl
            temp['Ticker'] = [s]*len(temp)
            temp['Date'] = pd.to_datetime(temp.index)
            if 'DateTime' not in temp.columns and 'Date' in temp.columns:
                temp.rename(columns={'Date':'DateTime'}, inplace=True)
            
            if 'Volume' not in temp.columns:
                temp['Volume'] = [0]*len(temp)
            
            # Calculate signal and apply filter
            temp = signal(df=temp, strat_name=strat)
            if apply_filter and strategies[strat].filter != None and 'MA' in strategies[strat].filter:
                period = int(strategies[strat].filter.split('_')[-1])
                temp['filter'] = temp['Close'].rolling(period).mean()
                temp[temp.columns[-2]] = np.where(temp['Close'] > temp['filter'], temp[temp.columns[-2]], 0)


            # Data treatment
            temp.sort_values('DateTime', inplace=True)
            temp['DateTime'] = pd.to_datetime(temp['DateTime'], unit='s')
            if temp['DateTime'].iloc[0].tzinfo != None:
                temp['DateTime'] = temp['DateTime'].dt.tz_convert(None)
            temp['Close'] = temp['Close'].fillna(method='ffill')
            temp['Spread'] = temp['Spread'].fillna(method='ffill')
            temp['Open'] = np.where(temp['Open'] != temp['Open'], temp['Close'], temp['Open'])
            temp['High'] = np.where(temp['High'] != temp['High'], temp['Close'], temp['High'])
            temp['Low'] = np.where(temp['Low'] != temp['Low'], temp['Close'], temp['Low'])

            # Store data
            if t in data:
                for c in temp:
                    if c not in data[t]:
                        data[t][c] = temp[c]
            else:
                data[t] = temp.copy()

            if portfolio:
                # Store portfolio
                if t not in portfolio:
                    portfolio[t] = []
            
                if temp.iloc[-1][f'{strat}Entry'] > 0:
                    trade = Trade(data=temp, signal='long', strategy=strategies[strat], 
                                balance=getEquity())
                    portfolio[t].append({**trade.to_dict(), **{'trade':trade}})

    if get_portfolio:
        return data, portfolio
    else:
        return data

def testTradeFiles():

    files = TradesFiles()
    trades = files._generateTrades()
    files.addOpenTrades(trades, mode='w')
    open = files.getOpenTrades()
    print(len(open))
    print(open.iloc[4])
    files.deleteOpenTrade(open.iloc[4])
    open = files.getOpenTrades()
    print(len(open))

if __name__ == '__main__':

    dg = DeGiro('OneMade','Onemade3680')
    signals = Signals(backtest=True, side=Signals.Side.LONG, errors=False, verbose=False)
    indicators = Indicators(errors=False)
    files = TradesFiles(path=trades_url)
    executions = []


    # Prepare data needed for backtest
    data, portfolio = getData(strategies, get_portfolio=True)

    # Execute orders
    for symbol in portfolio:

        df = pd.DataFrame(portfolio[symbol])
        initial_size = df['size'].sum()
        df['size'] = np.where(df['signal'] == 'short', -df['size'], df['size'])

        total_size = df['size'].sum()
        if total_size > 0:
            side = 'long'
        elif total_size < 0:
            side = 'short'
        else:
            side = None

        if side != None:

            df = df[df['signal'] == side]
            df['size'] = np.where(df['signal'] == 'short', -df['size'], df['size'])
            for order in df.groupby('order'):

                size = order[1]['size'].sum()/initial_size * total_size 
                for entry in order['entry'].groupby('entry'):

                    # Check if there is enough balance 
                    balance = getEquity()
                    size = size if size*entry[0] <= balance else balance/entry[0] # Hay un problema con esto y el cierre de operaciones según las operaciones independientes

                    # Prepare trade
                    trade = entry[1]['trade'].iloc[-1]
                    postOrder(product_id=symbol, side=side[0], order=order[0], 
                                entry=entry, sl_dist=trade.sldist, size=size)
                    
                    time.sleep(random.randint(40, 90))
    
    # Save portfolio to opened csv
    df = pd.DataFrame(portfolio.values())
    files.addOpenTrades(df, file=open_trades_name, mode='a')

    # Continued execution
    loop = False
    while loop:

        if start_time < dt.datetime.today().time() and dt.datetime.today().time() < end_time:

            data = getData(strategies, get_portfolio=False)
            trades = files.getOpenTrades(file=open_trades_name)
            # TODO! Check the exits for all the trades

            time.sleep(random.randint(60,120))

        # If market closed close execution
        elif end_time <= dt.datetime.today().time():
            loop = False

        # If market has not opened wait till it opens
        elif dt.datetime.today().time() < start_time:
            for s in strategies:
                for a in strategies[s].assets:
                    quote = dg.getQuote(product_id=tickers[a][broker])['data']
                    if quote['tradingStartTime'] < start_time:
                        start_time = quote['tradingStartTime']
                    if end_time < quote['tradingEndTime']:
                        end_time = quote['tradingEndTime']
                        
            time.sleep( time_interval(dt.datetime.today().time(), start_time, seconds=True) )

        else:
            time.sleep(random.randint(60,120))