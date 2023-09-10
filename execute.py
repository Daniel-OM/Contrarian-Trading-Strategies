
import copy
import datetime as dt
import numpy as np
import pandas as pd
#from google_sheets.google_sheets import GoogleSheets
from backtest import StrategyConfig
from degiro import DeGiro, IntervalType, Product, ResolutionType, Order, DataType
from indicators import Indicators
from signals import Signals
from config import strategies, tickers, broker, apply_filter


class Trade:

    def __init__(self, data:pd.DataFrame, signal:str, strategy:StrategyConfig,
                 balance:float) -> None:

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
        
        candle = data.iloc[-1]
        strategy = copy.deepcopy(strategy)
        asset = copy.deepcopy(strategy.assets[candle['Ticker']])

        sldist = asset.sl * candle['distATR']
        tpdist = asset.tp * candle['distATR']

        self.datetime = candle['DateTime']
        self.entrytime =  dt.datetime.today().strftime('%Y-%m-%d %H:%M')
        self.exittime = None
        self.ticker = candle['Ticker']
        self.asset = asset
        self.strategy = strategy
        self.order = asset.order_type
        self.signal = signal
        self.entry = self.calculateEntry(candle, data.iloc[-2])
        self.exit = None
        self.sl = self.entry - sldist if signal == 'long' else self.entry + sldist
        self.tp = self.entry + tpdist if signal == 'long' else self.entry - tpdist
        self.sldist = sldist
        self.returns = None
        self.risk = asset.risk
        self.balance = balance
        self.size = self.calculateSize()
        self.result = None
        self.candle = candle

    def calculateEntry(self, candle:dict, prev_candle:dict) -> float:

        entry = candle['Open']
        if self.signal == 'long':
            if self.order == 'stop' and 'High' in prev_candle[candle['Ticker']]:
                    entry = prev_candle[candle['Ticker']]['High']
            elif self.order == 'limit' and 'Low' in prev_candle[candle['Ticker']]:
                    entry = prev_candle[candle['Ticker']]['Low']
        elif self.signal == 'short':
            if self.order == 'stop' and 'Low' in prev_candle[candle['Ticker']]:
                entry = prev_candle[candle['Ticker']]['Low']
            elif self.order == 'limit' and 'High' in prev_candle[candle['Ticker']]:
                entry = prev_candle[candle['Ticker']]['High']

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

        self.size = int(self.risk * self.balance / self.sldist)
        if self.size > self.asset.max_size:
            self.size = self.asset.max_size
        elif self.size < self.asset.min_size:
            self.size = self.asset.min_size

        if self.balance < 0:
            self.size = 0.0

        return self.size

    def exitTrade(self, exit:float, ) -> None:

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
    
    def to_dict(self):

        '''
        Generates a dictionary with the trade data.

        Returns
        -------
        object: dict
            Contains the data.
        '''

        self.exitTrade()

        return self.__dict__

def postOrder(trade:Trade) -> None:

    product = dg.searchProducts(trade.asset.id)[0]
    side = 'BUY' if trade.side == 'long' else 'SELL'
    
    if trade.order == 'stop':
        stop = trade.entry + trade.candle['SLdist']/2 if side == 'BUY' \
                else trade.entry - trade.candle['SLdist']/2
        dg.tradeOrder(product, trade.size, side, Order.Type.STOPLIMIT, 
                    Order.Time.GTC, limit=trade.entry, stop_loss=stop)
    elif trade.order == 'stoplimit':
        dg.tradeOrder(product, trade.size, side, Order.Type.STOPLIMIT, 
                    Order.Time.GTC, stop_loss=trade.entry)
    elif trade.order == 'limit':
        dg.tradeOrder(product, trade.size, side, Order.Type.LIMIT,
                    Order.Time.GTC, limit=trade.entry)
    else:
        dg.tradeOrder(product, trade.size, side, Order.Type.MARKET, 
                    Order.Time.GTC)


def enterOrders(trades:dict) -> None:

    for symbol in trades:
        df = pd.DataFrame(trades[symbol])
        product = dg.searchProducts(symbol)[0]

        for side in df.groupby('signal'):
            side = 'BUY' if side[0] == 'long' else 'SELL'
            temp = side[1]

        stop = trade.entry + trade.candle['SLdist']/2 if side == 'BUY' \
                else trade.entry - trade.candle['SLdist']/2
        if trade.order == 'stop':
            dg.tradeOrder(product, trade.size, side, Order.Type.STOPLIMIT, 
                        Order.Time.GTC, limit=trade.entry, stop_loss=stop)
        elif trade.order == 'limit':
            dg.tradeOrder(product, trade.size, side, Order.Type.LIMIT,
                        Order.Time.GTC, limit=trade.entry)
        else:
            dg.tradeOrder(product, trade.size, side, Order.Type.MARKET, 
                        Order.Time.GTC)
        
def getEquity() -> float:

    return float(dg.getData(DataType.CASHFUNDS)[0][3:])

def getPortfolio() -> list:

    return dg.getData(DataType.PORTFOLIO)

    

dg = DeGiro('OneMade','Onemade3680')
signals = Signals(backtest=True, side=Signals.Side.LONG, errors=False)
indicators = Indicators(errors=False)


# Prepare data needed for backtest
strategies = []
data = {}
portfolio = {}
for strat in strategies:

    # Prepare signal if exists
    if strat not in dir(signals):
        print(f'{strat} not between the defined signals.')
        continue
    signal = getattr(signals, strat)

    for t,c in strategies[strat].assets.items():
        
        strategies[strat].assets[t].id = tickers[t][broker]

        # Get data
        if t not in data:
            products = dg.searchProducts(tickers[t][broker])
            temp = dg.getCandles(Product(products[0]).id, resolution=ResolutionType.H1, 
                                    interval=IntervalType.Y3)
        else:
            temp = data[t].copy()

        # Add columns
        temp.loc[:,'distATR'] = indicators.atr(n=20, method='s', dataname='ATR', new_df=temp)['ATR']
        temp.loc[:,'SLdist'] = temp['distATR'] * c.sl
        temp.loc[:,'Ticker'] = [t]*len(temp)
        temp.loc[:,'Date'] = pd.to_datetime(temp.index)
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
        temp.loc[:,'DateTime'] = pd.to_datetime(temp['DateTime'], unit='s')
        if temp['DateTime'].iloc[0].tzinfo != None:
            temp['DateTime'] = temp['DateTime'].dt.tz_convert(None)
        temp['Close'] = temp['Close'].fillna(method='ffill')
        temp['Spread'] = temp['Spread'].fillna(method='ffill')
        temp['Open'] = np.where(temp['Open'] != temp['Open'], temp['Close'], temp['Open'])
        temp['High'] = np.where(temp['High'] != temp['High'], temp['Close'], temp['High'])
        temp['Low'] = np.where(temp['Low'] != temp['Low'], temp['Close'], temp['Low'])

        t = tickers[t][broker]
        # Store data
        if t in data:
            for c in temp:
                if c not in data[t]:
                    data[t][c] = temp[c]
        else:
            data[t] = temp.copy()

        # Store portfolio
        if t not in portfolio:
            portfolio[t] = []
        
        if temp.iloc[-1][f'{strat}Entry'] > 0:
            trade = Trade(temp, 'long', strategies[strat], )
            portfolio[t].append(trade.to_dict())


# Execute orders