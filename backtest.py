
import os
import copy
import datetime as dt
import pytz
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signals import Signals
from indicators import OHLC, Indicators
#from google_sheets.google_sheets import GoogleSheets
from degiro import DeGiro, Product, IntervalType



class Commissions:

    def __init__(self, ctype:str='percentage', commission:float=5.0, 
                 cmin:float=1, cmax:float=None) -> None:

        '''
        Generate commissions configuration.

        Parameters
        ----------
        ctype: str
            Type of commissions input. It can be: 'percentage', 'perunit' or 'pershare'.
        commission: float
            Value of the commission. If the type is percentage it will be divided by 100.
        cmin: float
            Minimum value of the commission. Some brokers use minimum a dollar, here 
            it is represented.
        cmax: float
            Maximum value of the commission.
        '''

        self.type = 'perunit' if ctype == 'percentage' else ctype
        self.value = commission/100 if ctype == 'percentage' else commission
        self.min = cmin
        self.max = cmax
    
    def to_dict(self) -> dict:

        '''
        Generates a dictionary with the config for the commissions.

        Returns
        -------
        object: dict
            Contains the config for the commissions.
        '''

        return {
            'type': self.type,
            'value': self.value,
            'min': self.min,
            'max': self.max
        }

class AssetConfig:

    def __init__(self, name:str, risk:float=0.01, sl:float=None, tp:float=None, 
                 order:str='stop', min_size:float=1.0, max_size:float=10000.0, 
                 commission:Commissions=None) -> None:

        '''
        Generate commissions configuration.

        Parameters
        ----------
        name: str
            Name of the asset.
        risk: float
            Risk in per unit for the asset.
        sl: float
            ATR multiplier for the SL. If it's None then SL will be placed at 0 
            but shorts won't be available.
        tp: float
            ATR multiplier for the TP. If it's None then the trade will be 
            closed when a new entry is signaled.
        order: str
            Order type. It can be 'market', 'limit' or 'stop'.
        min_size: float
            Minimum size to trade the asset.
        max_size: float
            Maximum size available to trade the asset.
        commission: Commissions
            Commissions object associated to the asset, it depends on the asset.
        '''

        self.name = name
        self.risk = risk
        self.sl = sl
        self.tp = tp
        self.order_type = order
        self.min_size = min_size
        self.max_size = max_size
        self.commission = commission
    
    def to_dict(self) -> dict:

        '''
        Generates a dictionary with the config for the asset.

        Returns
        -------
        object: dict
            Contains the config for the asset.
        '''

        return {
            'name': self.name,
            'commission': self.commission, 
            'risk': self.risk,
            'SL': self.sl, 
            'TP': self.tp, 
            'order_type': self.order_type, 
            'min_size': self.min_size, 
            'max_size': self.max_size
        }

class StrategyConfig:

    def __init__(self, name:str, assets:dict={}, use_sl:bool=True, use_tp:bool=True, 
                 time_limit:int=50, timeframe:str='H1', filter:str=None) -> None:

        '''
        Generate commissions configuration.

        Parameters
        ----------
        name: str
            Name of the strategy.
        assets: dict[AssetConfig]
            Dictionary with the assets tradeable by the strategy.
        use_sl: float
            True to use SL as exit method. If the asset config has None as SL multiplier 
            attribute the strategy will only be able to long.
        use_tp: float
            True to use TP as exit method.
        time_limit: int
            Number of candles to wait for the trade to exit, after which the trade 
            will be manually closed.
        timeframe: float
            Minimum size to trade the asset.
        max_size: float
            Maximum size available to trade the asset.
        commission: Commissions
            Commissions object associated to the asset, it depends on the asset.
        '''

        self.name = name
        self.assets = assets
        self.use_sl = use_sl
        self.use_tp = use_tp
        self.time_limit = time_limit
        self.timeframe = timeframe
        self.filter = filter

    def addAsset(self, name:str, config:AssetConfig) -> None:

        '''
        Adds an asset to the dictionary of traded assets.
        '''

        self.assets[name] = config
    
    def to_dict(self) -> dict:

        '''
        Generates a dictionary with the config for the strategy.

        Returns
        -------
        object: dict
            Contains the config for the strategy.
        '''

        return {
            'name': self.name,
            'assets': {a: self.assets[a].to_dict() for a in self.assets.keys()},
            'use_sl': self.use_sl,
            'use_tp': self.use_tp,
            'time_limit': self.time_limit,
            'timeframe': self.timeframe,
            'filter': self.filter,
        }
    
class BtConfig:

    '''
    Class used to create the backtest config.
    '''

    def __init__(self, init_date:str=None, final_date:str=None, capital:float=10000.0,
                monthly_add:float=0, use_sl:bool=True, use_tp:bool=True, time_limit:int=365,
                min_size:float=1, max_size:float=10000000, commission:Commissions=None,
                max_trades:int=3, filter_ticker:bool=True, filter_strat:bool=False,
                reset_orders:bool=True, continue_onecandle=True, offset_aware:bool=False
                ) -> None:

        '''
        Generates the main config object for the backtest.

        Parameters
        ----------
        init_date: str
            Date to start the backtest. Must be in format: YYYY-MM-DD.
        final_date: str
            Date to end the backtest. Must be in format: YYYY-MM-DD.
        capital: float
            Starting capital for the backtest. Default is 10000.
        monthly_add: float
            Quantity to add to the balance every month.
        use_sl: bool
            True to use fixed SL. Default is True.
        use_tp: bool
            True to use fixed TP. Default is True.
        time_limit: int
            Number of maximum candles for an open trade before closing.
        min_size: float
            Minimum contracts for a position.
        max_size: float
            Maximum contracts for a position.
        commission: Commissions
            Commissions object associated to broker. This one will be added 
            and applied to each trade in each asset.
        max_trades: int
            Maximum trades open at the same time.
        filter_ticker: bool
            True to apply the max_trades to each ticker.
        filter_strat: bool
            True to apply the max_trades to each strategy.
        reset_orders: bool
            True to reset pending orders in a ticker if another one
            of the same direction (long or short) appeared.
        continue_onecandle: bool
            OneCandle trades are those where a candle triggered the entry, 
            the SL and the TP. As we don't know the order in which they 
            were triggered we can ignore the exit by setting the input to 
            True. This way the trade will stay open till a future candle 
            triggers another exit signal.
        offset_aware: bool
            True to give a timezone to the dates.
        '''

        if init_date == None:
            init_date = (dt.date.today() - dt.timedelta(days=365*2)).strftime('%Y-%m-%d')
        if final_date == None:
            final_date = dt.date.today().strftime('%Y-%m-%d')

        self.init_date = self._dateFormat(init_date, offset_aware)
        self.final_date = self._dateFormat(final_date, offset_aware)
        self.capital = capital
        self.monthly_add = monthly_add
        self.use_sl = use_sl
        self.use_tp = use_tp
        self.time_limit = time_limit
        self.min_size = min_size
        self.max_size = max_size
        self.commission = commission
        self.max_trades = max_trades
        self.filter_ticker = filter_ticker
        self.filter_strat = filter_strat
        self.reset_orders = reset_orders
        self.continue_onecandle = continue_onecandle

    def _dateFormat(self, datetime:str, offset:bool=False) -> dt.datetime:

        '''
        Generates a dictionary with the config for the backtest.

        Parameters
        ----------
        datetime: str
            DateTime to start the backtest. Must be in format: YYYY-MM-DD.
        offset: bool
            Make the date offset aware. The offset will be UTC.

        Returns
        -------
        datetime: str | dt.datetime
            Contains the formated date.
        '''
        
        if isinstance(datetime, str):
            datetime = dt.datetime.strptime(datetime, '%Y-%m-%d')
            if offset:
                datetime = datetime.replace(tzinfo=pytz.UTC)

        return datetime

    def to_dict(self) -> dict:

        '''
        Generates a dictionary with the config for the backtest.

        Returns
        -------
        object: dict
            Contains the config for the backtest.
        '''
        
        return {
            'capital': self.capital,
            'monthly': self.monthly_add,
            'init_date': self.init_date,
            'final_date': self.final_date,
            'use_sl': self.use_sl,
            'use_tp': self.use_tp,
            'time_limit': self.time_limit,
            'min_size': self.min_size,
            'max_size': self.max_size,
            'commission': self.commission,
            'maxtrades': self.max_trades,
        }

class Trade:

    def __init__(self, candle:dict, signal:str, strategy:StrategyConfig,
                 entry:float, balance:float) -> None:

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

        # comission = asset.commission.value * 100 if 'JPY' in candle['Ticker'] and \
        #             asset.commission.type == 'pershare' else asset.commission.value
        strategy = copy.deepcopy(strategy)
        asset = copy.deepcopy(strategy.assets[candle['Ticker']])

        sldist = asset.sl * candle['distATR']
        tpdist = asset.tp * candle['distATR']

        self.datetime = candle['DateTime']
        self.entrytime =  candle['DateTime']
        self.exittime = candle['DateTime']
        self.ticker = candle['Ticker']
        self.asset = asset
        self.strategy = strategy
        self.order = asset.order_type
        self.signal = signal
        self.entry = entry
        self.exit = candle['Close']
        self.sl = entry - sldist if signal == 'long' else entry + sldist
        self.tp = entry + tpdist if signal == 'long' else entry - tpdist
        self.sldist = sldist
        self.returns = candle['Close'] - entry
        self.spread = candle['Spread']
        self.commission_type = asset.commission
        self.commission = 0.0
        self.risk = asset.risk
        self.balance = balance
        self.size = self.calculateSize()
        self.result = self.returns * self.size - self.calculateCommission()
        self.high = candle['High']
        self.low = candle['Low']
        self.candles = []
        self.onecandle = False
        self.method = None

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
    
    def calculateReturns(self, entry:float=None, exit:float=None) -> float:

        '''
        Calculates the returns of the trade.

        Returns
        -------
        returns: float
            Returns of the trade.
        '''

        if entry != None:
            self.entry = entry
        if exit != None:
            self.exit = exit

        if self.signal == 'long':
            self.returns = self.exit - self.entry
        elif self.signal == 'short':
            self.returns = self.entry - self.exit
        else:
            raise(ValueError(f'Signal ({self.signal}) not valid.'))

        return self.returns
    
    def calculateCommission(self):

        '''
        Calculates the commission applied to the trade.

        Returns
        -------
        commission: float
            Commission charged for the trade.
        '''

        commission = self.commission_type.value * self.size \
                    if self.commission_type.type == 'pershare' \
                    else self.commission_type.value * self.calculateResult(com=False)
        
        if self.commission_type.max != None and \
            commission > self.commission_type.max:
            commission = self.commission_type.max
        elif self.commission_type.min != None and \
            commission < self.commission_type.min:
            commission = self.commission_type.min

        self.commission = commission

        return self.commission
    
    def calculateResult(self, com:bool=True):

        '''
        Calculates the result of the trade.

        Returns
        -------
        result: float
            Result of the trade.
        '''

        self.result = self.calculateReturns() * self.size
        if com:
            self.result -= self.calculateCommission()

        return self.result
    
    def tradeExited(self) -> None:

        '''
        Closes a trade and calculates all the final data.
        '''
        
        self.calculateSize()
        self.calculateReturns()
        self.calculateCommission()
        self.calculateResult()

    def to_dict(self):

        '''
        Generates a dictionary with the trade data.

        Returns
        -------
        object: dict
            Contains the data.
        '''

        return {
            'OrderTime': self.datetime,
            'EntryTime': self.entrytime,
            'ExitTime': self.exittime,
            'Ticker': self.ticker,
            'Strategy': self.strategy.to_dict(),
            'Order': self.order,
            'Signal': self.signal,
            'Entry': self.entry,
            'Exit': self.exit,
            'SL': self.sl,
            'TP': self.tp,
            'Return': self.calculateReturns(),
            'Result': self.calculateResult(),
            'Method': self.method,
            'Spread': self.spread,
            'Commission': self.calculateCommission(),
            'CommissionStruc': self.asset.commission.to_dict(),
            'Risk': self.risk,
            'Balance': self.balance,
            'Size': self.calculateSize(),
            'SLdist': self.sldist,
            'High': max([c['High'] for c in self.candles]+[0]),
            'Low': min([c['Low'] for c in self.candles]+[0]),
            'Candles': self.candles,
            'OneCandle': self.onecandle,
            'Asset': self.asset.to_dict()
        }

class KPIs:

    def __init__(self, df:pd.DataFrame) -> None:

        if not isinstance(df, pd.DataFrame):
            raise ValueError('No DataFrame was passed.')

        self.days = np.busday_count(df['DateTime'].tolist()[0].date(), df['DateTime'].tolist()[-1].date())
        self.frequency = len(df)/self.days * 100//1/100
            
        temp = df.copy()        
        temp['Ret'] = temp['Result']/(temp['SLdist']*temp['Size']) * temp['Risk']
        temp['CumRet'] = (1+temp['Ret']).cumprod()

        # Backtest analysis
        self.winrate = len(temp['Return'][temp['Return'] > 0.0])/len(temp['Return'])
        self.avg_win = temp['Ret'][temp['Ret'] > 0].mean()
        self.avg_loss = temp['Ret'][temp['Ret'] < 0].mean()
        self.expectancy = (self.winrate * self.avg_win - (1-self.winrate)*abs(self.avg_loss))
        self.kelly = self.expectancy/self.avg_win
        self.avg_risk = temp['Risk'].mean()
        self.balance = temp['Balance'].iloc[-1]
        self.max_dd = temp['AccountDD'].max()
        self.n_trades = len(temp)
        
    def to_dict(self) -> dict:

        return self.__dict__
    
class BackTest(OHLC):

    '''
    Class used to carry out the backtest of the strategy.
    '''
    
    config = BtConfig((dt.date.today() - dt.timedelta(days=365*2)).strftime('%Y-%m-%d'), 
                      dt.date.today().strftime('%Y-%m-%d'), capital=10000.0, monthly_add=200, 
                      use_sl=True, use_tp=True, time_limit=None, min_size=1000, 
                      max_size=10000000, commission=Commissions(), max_trades=3,
                      filter_ticker=True, filter_strat=False, reset_orders=True,
                      continue_onecandle=True)

    def __init__(self, strategies:dict, config:BtConfig=None) -> None:

        '''
        Initialize the Backtesting object.

        Parameters
        ----------
        strategies: dict
            Dictionary with the data for strategies and the pairs.
            Example:
            ex_dict = {
                'strategy1': StrategyConfig(),
                ...
            }
        config: BtConfig
            BtConfig object with the backtest config.
        '''

        self.strategies = strategies
        self.config = config if config != None else self.config

    def fillHalts(self, df:pd.DataFrame) -> pd.DataFrame:

        '''
        Fills halts.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame containing the candles data.

        Returns
        -------
        df: pd.DataFrame
            Contains all the DataFrame without the previous close value 
            for the halts.
        '''

        df['Close'] = df['Close'].fillna(method='ffill')
        df['Spread'] = df['Spread'].fillna(method='ffill')
        df['Open'] = np.where(df['Open'] != df['Open'], df['Close'], df['Open'])
        df['High'] = np.where(df['High'] != df['High'], df['Close'], df['High'])
        df['Low'] = np.where(df['Low'] != df['Low'], df['Close'], df['Low'])

        return df
    
    def getEntries(self, candle) -> None:

        '''
        Gets the column names of the different Entry strategies.

        Parameters
        ----------
        candle: 
            Contains the current iteration candle.
        '''
        
        self.entries = [c.replace('Entry','').replace('entry','') for c in candle.index \
                        if 'entry' in c.lower()]
    
    def getExits(self, candle) -> None:

        '''
        Gets the column names of the different Exit strategies.

        Parameters
        ----------
        candle: 
            Contains the current iteration candle.
        '''
        
        self.exits = [c.replace('Exit','').replace('exit','') for c in candle.index \
                        if 'exit' in c.lower()]

    def openQty(self, candle, open_trades:list, filter_ticker:bool=True, 
                filter_strat:bool=False) -> dict:

        '''
        Gets the quantity of open trades.

        Parameters
        ----------
        candle: 
            Contains the current iteration candle.
        open_trades: list
            List with the dictionary that contains the current open trades.
        filter_ticker: bool
            True to return a dictionary with the open trades for each symbol.
        filter_strat: bool
            True to return a dictionry with the open trades for each strategy.

        Returns
        -------
        data_df: int | dict
            Contains the candles of all the pairs orderes by Date with 
            the correct format.
        '''

        qty = {}
        # Filtered by ticker and strat
        if filter_ticker and filter_strat:

            # Store the trade qty for all the open trades
            for trade in open_trades:
                # If new ticker
                if trade.ticker not in qty:
                    qty[trade.ticker] = {trade.strategy: 1}
                # For already added tickers
                elif trade.strategy not in qty[trade.ticker]:
                    qty[trade.ticker][trade.strategy] = 1
                else:
                    qty[trade.ticker][trade.strategy] += 1

            # If the current iteration ticker has no open trades add it
            if candle['Ticker'] not in qty:
                qty[candle['Ticker']] = {}
                for strat in self.entries:
                    qty[candle['Ticker']][strat] = 0

            for strat in self.entries:
                if strat not in qty[candle['Ticker']]:
                    qty[candle['Ticker']][strat] = 0

        # Filtered by ticker
        elif filter_ticker:

            # Store the trade qty for all the open trades
            for trade in open_trades:
                # If new ticker
                if trade.ticker not in qty:
                    qty[trade.ticker] = 1
                # For already added tickers
                else:
                    qty[trade.ticker] += 1

            # If the current iteration ticker has no open trades add it
            if candle['Ticker'] not in qty:
                qty[candle['Ticker']] = 0

        # Filtered by strat
        elif filter_strat and not filter_ticker:

            # Store the values for all the open trades
            for trade in open_trades:
                if trade.strategy not in qty:
                    qty[trade.strategy] = 1
                else:
                    qty[trade.strategy] += 1

            # If the current iteration strategies have no open trades add them
            for strat in self.entries:
                if strat not in qty:
                    qty[strat] = 0

        # Not filtered
        else:
            qty = len(open_trades)
        
        return qty

    def currentCapital(self, balance:float, open_trades:list) -> float:

        '''
        Calculate current available capital.

        Parameters
        ----------
        balance: float
            Current accoutn balance.
        open_trades: list
            List of currently open trades.

        Returns
        -------
        capital: float
            Available capital to trade.
        '''

        capital = balance
        for trade in open_trades:
            capital -= trade.size * trade.entry

        return capital

    def backtest(self, df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Carries out the backtest logic.

        Parameters
        ----------
        df: pd.DataFrame
            Contains the complete candle data.

        Returns
        -------
        trades: pd.DataFrame
            Contains the trades carried out during the backtest.
        '''

        # Check if the needed data is in the dataframe
        self.data_df = self._newDf(df, needed_cols=['DateTime', 'Open', 'High', 'Low', 'Close', 'Spread', 'distATR'], 
                                   overwrite=True)

        # Initialize variables
        last_date = None
        open_trades = []
        open_orders = []
        closed_trades = []
        prev_candle = {}
        for t in self.strategies:
            prev_candle[t] = {}
        balance = [self.config.capital]
        self.getEntries(self.data_df.iloc[0])
        self.getExits(self.data_df.iloc[0])

        # Group data by DateTime
        for g in self.data_df.groupby('DateTime'):
          
            date_result = 0
            current_date = g[0]

            # Iterate for each asset in this DateTime
            for i in g[1].index:

                candle = g[1].loc[i]

                if candle['distATR'] != candle['distATR']:
                    continue
                
                # Check if we are between the backtest dates
                if candle['DateTime'] < self.config.init_date or candle['DateTime'] > self.config.final_date:
                    continue

                # Add the monthly add if greater than 0 and the month has changed
                if self.config.monthly_add > 0 and last_date != None and \
                    last_date.month != current_date.month:
                    balance[-1] = balance[-1] + self.config.monthly_add
                    
                # Look for entries
                if len(self.entries) > 0:

                    for strat in self.entries:

                        # Get trades qty
                        trades_qty = self.openQty(candle, open_trades, self.config.filter_ticker, self.config.filter_strat)
                        if self.config.filter_ticker and self.config.filter_strat and strat in trades_qty[candle['Ticker']]:
                            trades_qty = trades_qty[candle['Ticker']][strat]
                        elif self.config.filter_ticker:
                            trades_qty = trades_qty[candle['Ticker']]
                        elif self.config.filter_strat:
                            trades_qty = trades_qty[strat]

                        # If there are any orders and didn't reach the trades qty limit
                        if candle[f'{strat}Entry'] != 0 and trades_qty < self.config.max_trades:

                            asset = self.strategies[strat].assets[candle['Ticker']]

                            # Long orders
                            if candle[f'{strat}Entry'] > 0:
                                # Buy order entry price
                                if asset.order_type == 'market':
                                    entry = candle['Open'] + candle['Spread']
                                elif asset.order_type == 'stop':
                                    entry = prev_candle[candle['Ticker']]['High']
                                elif asset.order_type == 'limit':
                                    entry = prev_candle[candle['Ticker']]['Low']

                                # Check if the trade is already open
                                entered = False
                                for t in open_trades:
                                    if t.entry == entry or candle['Open'] == prev_candle[candle['Ticker']]['Open']:
                                        entered = True
                                        break

                                # Reset long open orders
                                if self.config.reset_orders:
                                    open_orders = [order for order in open_orders if (order.signal != 'long') or \
                                                   (order.ticker != candle['Ticker']) or (order.strategy != strat)]

                                # Define the new buy order
                                trade = Trade(candle, 'long', self.strategies[strat], entry, balance[-1])
                                
                                if not entered:
                                    current_capital = self.currentCapital(balance[-1], open_trades)
                                    if trade.entry * trade.size >= current_capital:
                                        trade.calculateSize(balance=current_capital)

                                    # If market order execute it if not append to orders
                                    if asset.order_type == 'market':
                                        open_trades.append(trade)
                                    else:
                                        open_orders.append(trade)

                            # Short orders
                            if candle[f'{strat}Entry'] < 0:
                                # Sell order entry price
                                if asset.order_type == 'market':
                                    entry = candle['Open']# - candle['Spread']
                                elif asset.order_type == 'stop':
                                    entry = prev_candle[candle['Ticker']]['Low']
                                elif asset.order_type == 'limit':
                                    entry = prev_candle[candle['Ticker']]['High']

                                # Check if the trade is already open
                                entered = False
                                for t in open_trades:
                                    if t.entry == entry or candle['Open'] == prev_candle[candle['Ticker']]['Open']:
                                        entered = True
                                        break
                                    
                                # Reset long open orders
                                if self.config.reset_orders:
                                    open_orders = [order for order in open_orders if (order.signal != 'short') or \
                                                   (order.ticker != candle['Ticker']) or (order.strategy != strat)]

                                # Define the new sell order
                                trade = Trade(candle, 'short', self.strategies[strat], entry, balance[-1])
                                if not entered:
                                    current_capital = self.currentCapital(balance[-1], open_trades)
                                    if trade.entry * trade.size >= current_capital:
                                        trade.calculateSize(balance=current_capital)

                                    # If market order execute it if not append to orders
                                    if asset.order_type == 'market':
                                        open_trades.append(trade)
                                    else:
                                        open_orders.append(trade)

                # Review pending orders execution
                if len(open_orders) > 0:

                    delete = []
                    for order in open_orders:

                        if order.ticker == candle['Ticker']:

                            # STOP orders
                            if order.order == 'stop':
                          
                                if order.signal == 'long':

                                    if order.entry <= candle['High'] + order.spread:
                                        order.entry = order.entry if candle['Open'] < order.entry else candle['Open']
                                        order.entrytime = candle['DateTime']
                                        order.balance = balance[-1]
                                        open_trades.append(order)
                                        delete.append(order)

                                    elif candle['Low'] < order.sl:
                                        #print(f"Buy Stop Cancelled by SL: \n Open - {candle['Open']} Low - {candle['Low']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)
                                    elif candle['High'] > order.tp:
                                        #print(f"Buy Stop Cancelled by TP: \n Open - {candle['Open']} High - {candle['High']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)

                                if order.signal == 'short':

                                    if order.entry >= candle['Low']:
                                        order.entry = order.entry if candle['Open'] > order.entry else candle['Open']
                                        order.entrytime = candle['DateTime']
                                        order.balance = balance[-1]
                                        open_trades.append(order)
                                        delete.append(order)

                                    elif candle['High'] > order.sl:
                                        #print(f"Sell Stop Cancelled by SL: \n Open - {candle['Open']} High - {candle['High']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)
                                    elif candle['Low'] < order.tp:
                                        #print(f"Sell Stop Cancelled by TP: \n Open - {candle['Open']} Low - {candle['Low']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)
                                        
                            # LIMIT orders
                            elif order.order == 'limit':

                                if order.signal == 'long':

                                    if order.entry > candle['Low'] + order.spread:
                                        order.entry = order.entry if candle['Open'] > order.entry else candle['Open']
                                        order.entrytime = candle['DateTime']
                                        order.balance = balance[-1]
                                        open_trades.append(order)
                                        delete.append(order)

                                    elif candle['Low'] < order.sl:
                                        #print(f"Buy Limit Cancelled by SL: \n Open - {candle['Open']} Low - {candle['Low']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)
                                    elif candle['High'] > order.tp:
                                        #print(f"Buy Limit Cancelled by TP: \n Open - {candle['Open']} High - {candle['High']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)

                                if order.signal == 'short':

                                    if order.entry < candle['High']:
                                        order.entry = order.entry if candle['Open'] < order.entry else candle['Open']
                                        order.entrytime = candle['DateTime']
                                        order.balance = balance[-1]
                                        open_trades.append(order)
                                        delete.append(order)
                                        
                                    elif candle['High'] > order.sl:
                                        #print(f"Sell Limit Cancelled by SL: \n Open - {candle['Open']} High - {candle['High']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)
                                    elif candle['Low'] < order.tp:
                                        #print(f"Sell Limit Cancelled by TP: \n Open - {candle['Open']} Low - {candle['Low']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)
                    
                    # Delete from open orders if already executed
                    for d in delete:
                        open_orders.remove(d)

                # Store trade evolution
                for trade in open_trades:

                    if trade.ticker == candle['Ticker']:
                        trade.candles.append({'DateTime':candle['DateTime'] ,'Open':candle['Open'], 'High':candle['High'], 
                                        'Low':candle['Low'], 'Close':candle['Close'], 'Volume': candle['Volume']})
                        
                # Check open trades limits orders
                if len(open_trades) > 0:
                    
                    delete = []
                    for trade in open_trades:

                        if candle['Ticker'] == trade.ticker:
                            
                            exited = False

                            # Check SL
                            if self.config.use_sl and (not self.config.continue_onecandle or len(trade.candles) > 1): # trade.order != 'stop'

                                if trade.signal == 'short' and candle['High'] + trade.spread >= trade.sl: # High
                                    trade.exit = trade.sl if candle['Open'] < trade.sl else candle['Open']
                                    trade.method = 'SL'
                                    exited = True
                                    if len(trade.candles) <= 1 and candle['Low'] + trade.spread <= trade.tp and \
                                        candle['High'] + trade.spread >= trade.sl:
                                        trade.onecandle = True

                                if trade.signal == 'long' and candle['Low'] <= trade.sl: # Low
                                    trade.exit = trade.sl if candle['Open'] > trade.sl else candle['Open']
                                    trade.method = 'SL'
                                    exited = True
                                    if len(trade.candles) <= 1 and candle['High'] >= trade.tp and candle['Low'] <= trade.sl:
                                        trade.onecandle = True
                            
                            # Check TP
                            if self.config.use_tp and not exited and (not self.config.continue_onecandle or len(trade.candles) > 1): # trade.order != 'limit'

                                if trade.signal == 'short' and candle['Low'] + trade.spread <= trade.tp: #Low
                                    trade.exit = trade.tp if candle['Open'] > trade.tp else candle['Open']
                                    trade.method = 'TP'
                                    exited = True
                                    if len(trade.candles) <= 1 and candle['Low'] + trade.spread <= trade.tp and \
                                        candle['High'] + trade.spread >= trade.sl:
                                        trade.onecandle = True

                                if trade.signal == 'long' and candle['High'] >= trade.tp: #High
                                    trade.exit = trade.tp if candle['Open'] < trade.tp else candle['Open']
                                    trade.method = 'TP'
                                    exited = True
                                    if len(trade.candles) <= 1 and candle['High'] >= trade.tp and candle['Low'] <= trade.sl:
                                        trade.onecandle = True
                            
                            # Check time limit
                            if not exited and trade.strategy.time_limit > 0 and len(trade.candles) >= trade.strategy.time_limit:
                                trade.exit = candle['Close']
                                trade.method = 'TimeLimit'
                                exited = True

                            if exited:
                                trade.exittime = candle['DateTime']
                                trade.tradeExited()
                                closed_trades.append(trade)
                                date_result += trade.calculateResult()
                                delete.append(trade)
                    
                    # Delete open trades if already exited
                    for d in delete:
                        open_trades.remove(d)

                # Check open trades Exits if the strategy has exit conditions
                if len(self.exits) > 0:

                    delete = []
                    for strat in self.entries:

                        for trade in open_trades:
                            if trade.ticker == candle['Ticker'] and trade.strategy == strat:

                                exited = False

                                # Exit Buy
                                if candle[f'{strat}Exit'] == 1 and trade.signal == 'long':
                                    trade.exit = candle['Open']
                                    trade.method = 'Exit'
                                    exited = True
                                # Exit Sell
                                elif candle[f'{strat}Exit'] == -1 and trade.signal == 'short':
                                    trade.exit = candle['Open'] + trade.spread
                                    trade.method = 'Exit'
                                    exited = True
                                        
                                if exited:
                                    trade.exittime = candle['DateTime']
                                    trade.tradeExited()
                                    closed_trades.append(trade)
                                    date_result += trade.calculateResult()
                                    delete.append(trade)
                    
                    for d in delete:
                        open_trades.remove(d)

                last_date = copy.deepcopy(current_date)
                prev_candle[candle['Ticker']] = candle

            balance.append(balance[-1]+date_result)

        # Calculate and store final data
        # self.trades = self.tradesDF(closed_trades)
        self.closed_trades = closed_trades
        self.trades = self.tradesDF(pd.DataFrame(
                    [{**t.to_dict(), **{'Trades':t}} for t in closed_trades]))
        if not self.trades.empty:
            self.trades['StratName'] = self.trades['Strategy'].apply(lambda x: x['name'])
        # self.trades['WeekDay'] = self.trades['EntryTime'].dt.day_name()
        self.open_trades = pd.DataFrame(
                    [{**t.to_dict(), **{'Trades':t}} for t in open_trades])
        self.open_orders = pd.DataFrame(
                    [{**t.to_dict(), **{'Trades':t}} for t in open_orders])

        return self.trades.copy()

    def tradesDF(self, trades:list) -> pd.DataFrame:

        if not isinstance(trades, pd.DataFrame):
            size = []
            result = []
            balance = [self.config.capital]
            last_date = None
            trades = [copy.deepcopy(t) for t in trades]
            for trade in trades:
                if self.config.monthly_add > 0 and last_date != None and \
                    trade.datetime.month != last_date.month:
                    balance[-1] = balance[-1] + self.config.monthly_add
                trade.balance = balance[-1]
                size.append(trade.calculateSize(balance=balance[-1]))
                result.append(trade.calculateResult())
                balance.append(balance[-1] + trade.result)
                last_date = trade.datetime
            trades = pd.DataFrame([t.to_dict() for t in trades])

        if not trades.empty:
            trades['InitBalance'] = trades['Balance'].copy()
            trades['Balance'] = trades['Balance'] + trades['Result']
            trades['RetPct'] = trades['Result'] / trades['InitBalance']
            trades.loc[:,'AccountPeak'] = trades['Balance'].cummax()
            trades.loc[:,'AccountDD'] = 1 - trades['Balance']/trades['AccountPeak']

        return trades

    def calculateBalance(self, trades:list=None, verbose:bool=True) -> pd.DataFrame:

        if not isinstance(trades, list):
            if self.trades.empty:
                print('There are no trades to plot')
                return None
            trades = self.trades['Trades'].copy().to_list()

        trades = copy.deepcopy(trades)
        balance = []
        for t,trade in enumerate(trades):
            result = []
            prev_c = trade.candles[0]
            for i in range(len(trade.candles)):
                c = trade.candles[i]
                if len(trade.candles) == 1:
                    val = ((trade.exit - trade.entry) if trade.signal == 'long' \
                        else (trade.entry - trade.exit)) * trade.size -trade.calculateCommission()
                elif i == 0:
                    val = ((c['Close'] - trade.entry) if trade.signal == 'long' \
                        else (trade.entry - c['Close'])) * trade.size -trade.calculateCommission()
                elif i == len(trade.candles)-1:
                    val = ((trade.exit - prev_c['Close']) if trade.signal == 'long' \
                        else (prev_c['Close'] - trade.exit)) * trade.size
                else:
                    val = ((c['Close'] - prev_c['Close']) if trade.signal == 'long' \
                        else (prev_c['Close'] - c['Close'])) * trade.size

                balance.append({'DateTime':c['DateTime'], 'DailyBalance': val})
                result.append(balance[-1]['DailyBalance'])
                prev_c = c
            if verbose and sum(result) != trade.calculateResult():
                print(t)
                print(result, sum(result), trade.result)

        balance = pd.DataFrame(balance).groupby('DateTime').agg('sum')
        balance.reset_index(drop=False, inplace=True)
        balance.loc[:,'Added'] = np.where(balance['DateTime'].dt.month != \
                                          balance['DateTime'].shift(1).dt.month, 
                                          self.config.monthly_add, 0)
        balance.loc[:,'Added'] = balance['Added'].cumsum()
        balance.loc[:,'Balance'] = balance['DailyBalance'].cumsum() + self.config.capital
        balance.loc[:,'Balance'] = balance['Balance'] + balance['Added']
        balance.loc[:,'RetPct'] = balance['Balance'] / balance['Balance'].shift(1) - 1
        balance.loc[:,'AccountPeak'] = balance['Balance'].cummax()
        balance.loc[:,'AccountDD'] = 1 - balance['Balance']/balance['AccountPeak']

        return balance

    def saveResult(self, file:str='TradesBacktested.xlsx', sheet:str='CompleteTrades'):

        writer = pd.ExcelWriter(file)
        self.trades.to_excel(writer, sheet_name=sheet, index=False)
        self.calculateBalance().to_Excel(writer, sheet_name='Balance', index=False)
        writer.save()

    # def resultsToGoogle(self, sheetid:str, sheetrange:str):

    #     google = GoogleSheets(sheetid, secret_path=os.path.join('google_sheets','client_secrets.json'))
    #     google.appendValues(self.trades.values, sheetrange=sheetrange)

    def btKPI(self, df:pd.DataFrame=None, print_stats:bool=False) -> dict:

        days = np.busday_count(self.data_df['DateTime'].tolist()[0].date(), self.data_df['DateTime'].tolist()[-1].date())

        if not isinstance(df, pd.DataFrame):
            df = self.trades.copy()
            
        temp = df.copy()        
        temp['Ret'] = temp['Result']/(temp['SLdist']*temp['Size']) * temp['Risk']
        temp['CumRet'] = (1+temp['Ret']).cumprod()

        # Backtest analysis
        winrate = len(temp['Return'][temp['Return'] > 0.0])/len(temp['Return'])
        avg_win = temp['Ret'][temp['Ret'] > 0].mean()
        avg_loss = temp['Ret'][temp['Ret'] < 0].mean()
        expectancy = (winrate*avg_win - (1-winrate)*abs(avg_loss))

        stats = {
            'Winrate': winrate,
            'AvgWin': avg_win,
            'AvgLoss': avg_loss,
            'Expectancy': expectancy,
            'Days': days,
            'Frequency': len(temp)/days * 100//1/100,
            'Kelly': (winrate*avg_win - (1-winrate)*abs(avg_loss))/avg_win,
            'Avg risk': temp['Risk'].mean(),
            'BtBalance': temp['Balance'].tolist()[-1],
            'MaxDD': temp['AccountDD'].max(),
            '#trades': len(temp)
        }

        if print_stats:
            print(f"Winrate: {winrate :%}") # With two decimal spaces and commas for the thousands :,.2f
            print(f"Avg. Win: {avg_win :%}")
            print(f"Avg. Loss: {avg_loss :%}")
            print(f"Expectancy: {expectancy :%}")
            print(f"Trading frequency: {stats[strat]['Frequency']}")
            #print(f"Monthly Expectancy: {((1 + expectancy*len(trades)/days)**(20) - 1) :%}")
            #print(f"Anual Expectancy: {((1 + expectancy*len(trades)/days)**(52*5) - 1) :%}")
            print(f"Kelly: {stats[strat]['Kelly'] :%}")
            print(f"Backtest Max. DD: {stats[strat]['MaxDD'] :%}")
            # print(f'Ulcer Index: {(((trades['AccountDD'] * 100)**2).sum()/len(trades['AccountDD']))**(1/2) * 100//1/100}')

        return stats

    def weekDayKPI(self, trades:pd.DataFrame=None, df:bool=False):

        trades = self.trades if trades == None else trades

        trades['Date'] = pd.to_datetime(trades['EntryTime'], format='%Y-%m-%d %H:%M:%S')
        trades['WeekDay'] = trades['Date'].dt.day_name()
        day_stats = {}
        for g in trades.groupby('WeekDay'):

            day = g[1]
            temp = {}
            temp['winrate'] = len(day['Return'][day['Return'] > 0])/len(day['Return'])
            temp['avg_win'] = day['%ret'][day['%ret'] > 0].mean()
            temp['avg_loss'] = day['%ret'][day['%ret'] < 0].mean()
            temp['expectancy'] = (temp['winrate']*temp['avg_win'] - (1-temp['winrate'])*abs(temp['avg_loss']))
            temp['kelly'] = temp['expectancy']/temp['avg_win']
            day_stats[g[0]] = temp

        return pd.DataFrame(day_stats) if df else day_stats

    def tradesAnalysis(self, trades:pd.DataFrame=None, plot:bool=False):

        trades = self.trades if trades == None else trades

        temp_df = []
        patterns = []
        for i,t in enumerate(trades.values):
            temp = pd.DataFrame(t[16])
            if len(temp['Close']) > 1:
                temp['ret'] = (1 + temp['Close'].pct_change(1)).cumprod() - 1
            else:
                temp['ret'] = temp['Close']/temp['Open'] - 1
            
            if t[2] == 'Sell':
                temp['ret'] = -1*temp['ret']
                
            temp['cumMax'] = temp['ret'].cummax()
            max_idx = temp['cumMax'].idxmax()
            temp['cumMin'] = temp['ret'].cummin()
            min_idx = temp['cumMin'].idxmin()
            temp_df.append(temp)
            patterns.append({'Max': temp['ret'].max(), 'MaxIdx': temp['ret'].idxmax(),
                            'Min': temp['ret'].min(), 'MinIdx': temp['ret'].idxmin(),
                            'End': temp['ret'].tolist()[-1], 'EndIdx': len(temp['ret'])})

        if plot:
            temp_df = []

            fig = make_subplots(rows=1, cols=1)
            for i,t in enumerate(patterns):
                fig.add_trace(go.Scatter(x=[n for n in range(len(t['ret']))], y=t['ret'], name=f'Trade {i}'), row=1, col=1)
                fig.update_yaxes(title_text='Return', row=1, col=1)
                fig.update_xaxes(title_text='Candle since entry', row=1, col=1)
                fig.update_layout(title='Trade evolution', autosize=False,
                                    xaxis_rangeslider_visible=False,
                                    width=1000,
                                    height=700)

            fig.show()

        turn_loss = 0; turn_loss_idx = 0
        loss = 0; loss_idx = 0
        turn_win = 0; turn_win_idx = 0
        win = 0; win_idx = 0
        real = []
        for t in patterns:
            if t['Max'] == t['Max']:
                real.append(t)

                if t['MinIdx'] > t['MaxIdx'] and t['Max'] > 0:
                    turn_loss += 1
                    turn_loss_idx += t['MaxIdx']
                if t['Max'] <= 0:
                    loss += 1
                    loss_idx += t['MaxIdx']
                if t['MaxIdx'] > t['MinIdx'] and t['Min'] < 0:
                    turn_win += 1
                    turn_win_idx += t['MinIdx']
                if t['Min'] >= 0:
                    win += 1
                    win_idx += t['MinIdx']

        print(f'Avg. Index give up win {turn_loss_idx/turn_loss}')
        print(f'Prob. of give up win {turn_loss/len(real)}')
        print(f'Avg. Index straight loss {loss_idx/loss}')
        print(f'Prob of straight loss {loss/len(real)}')
        print(f'Avg. Index turn to win {turn_win_idx/turn_win}')
        print(f'Prob of turn to win {turn_win/len(real)}')
        print(f'Avg. Index straight win {win_idx/win}')
        print(f'Prob of straight win {win/len(real)}')

        return trades

    def btPlot(self, balance:bool=False, log:bool=True):

        if self.trades.empty:
            print('There are no trades to plot')
            return None
        
        if balance:
            balance = self.calculateBalance(verbose=False)
        else:
            balance = self.trades.copy()

        if 'DateTime' not in balance and 'ExitTime' in balance:
            balance['DateTime'] = balance['ExitTime'].copy()

        self.balance = balance.copy()
        # Plot Backtest results
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.0,
                            row_heights=[3,1],
                            specs=[[{'secondary_y': True}],[{'secondary_y': False}]])

        fig.add_trace(go.Scatter(x=balance['DateTime'], y=balance['Balance'], name='Balance'), 
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=balance['DateTime'], y=balance['AccountDD'] * 10000//1/100, 
                                 fill='tozeroy', name='DrawDown'), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=balance['DateTime'], y=balance['AccountPeak'], name='MaxBalance'), 
                      row=1, col=1, secondary_y=False)
        
        # Add Buy & Hold comparison
        fig.add_trace(go.Scatter(x=self.data_df['DateTime'], 
                      y=self.data_df['Close']/self.data_df['Close'].iloc[0]*self.config.capital, 
                      name='Buy & Hold'), 
                      row=1, col=1, secondary_y=False)

        fig.update_yaxes(title_text='Return ($)', row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text='DrawDown (%)', row=1, col=1, secondary_y=True)

        if log:
            fig.update_yaxes(type="log", row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(x=balance['DateTime'][balance['RetPct'] > 0.0],
                                 y=balance['RetPct'][balance['RetPct'] > 0.0] * 10000//1/100, 
                                 name='Wins', marker_color='green', mode='markers'), row=2, col=1)
        fig.add_trace(go.Scatter(x=balance['DateTime'][balance['RetPct'] <= 0.0], 
                                 y=balance['RetPct'][balance['RetPct'] <= 0.0] * 10000//1/100, 
                                 name='Losses', marker_color='red', mode='markers'), row=2, col=1)

        fig.update_yaxes(title_text='Return (%)', row=2, col=1)
            
        # fig.add_trace(go.Histogram(x=trades['%ret'][trades['%ret'] > 0], name='Wins', 
        #                           marker_color='green'), row=3, col=1)
        # fig.add_trace(go.Histogram(x=trades['%ret'][trades['%ret'] <= 0], name='Losses', 
        #                           marker_color='red'), row=3, col=1)

        # if log:
        #     fig.update_yaxes(type="log", row=3, col=1)

        # fig.update_yaxes(title_text='Qty.', row=3, col=1)
        # fig.update_xaxes(title_text='Return (%)', row=3, col=1)

        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_layout(title=f"Account balance {self.config.capital + self.trades['Result'].sum()*100//1/100}$", 
                          autosize=False, width=1000, height=700)

        fig.show()
    
    def stats(self, column:str='Ticker', plot:bool=False, 
                    print_stats:bool=False) -> pd.DataFrame:

        self.stats_dict = {}
        pairs = []
        trades = copy.deepcopy(self.trades)
        for g in trades.groupby(column):

            temp = g[1].copy()
            temp = self.tradesDF(trades=temp['Trades'].tolist())

            self.stats_dict[g[0]] = self.btKPI(df=temp, print_stats=print_stats)
            
            if plot:
                fig = make_subplots(specs=[[{'secondary_y': True}]])
                fig.add_trace(go.Scatter(x=temp['Date'],y=temp['Balance'], name='Balance'), 
                              secondary_y=False)
                fig.add_trace(go.Scatter(x=temp['Date'], y=temp['AccountDD'] * 10000//1/100, 
                                         fill='tozeroy', name='DrawDown'), secondary_y=True)

                fig.update_xaxes(title_text='Date')
                fig.update_yaxes(title_text='Return ($)', secondary_y=False)
                fig.update_yaxes(title_text='DrawDown (%)', secondary_y=True)
                fig.update_layout(title=f"{g[0]} account balance {self.config.capital + temp['Result'].sum()*100//1/100}$", 
                                autosize=False,width=1000,height=700,)
                fig.show()

            pairs.append(g[0])

        # Comparison of pairs and total
        stats_df = pd.DataFrame(self.stats_dict).T
        stats_df.sort_values(by=['Kelly'], ascending=False, inplace=True)

        return stats_df


if __name__ == '__main__':

    import yfinance as yf

    config = BtConfig('2010-01-01', dt.date.today().strftime('%Y-%m-%d'), 
                      capital=5000.0, monthly_add=200,  # (dt.date.today() - dt.timedelta(days=250)).strftime('%Y-%m-%d')
                      use_sl=True, use_tp=True, time_limit=None, min_size=1000, 
                      max_size=10000000, commission=Commissions(), max_trades=1000, 
                      filter_ticker=False, filter_strat=False, reset_orders=True,
                      continue_onecandle=False, offset_aware=False)


    strategies = {
        'trendExplosion': StrategyConfig(name='TE', assets={
            #'BTCUSD': {'commission':100.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':0.0001, 'max_size':10000000},
            #'ETHUSD': {'commission':1.0, 'risk': 0.02, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':0.0001, 'max_size':10000000},
            'GBP_JPY': AssetConfig(name='GBP_JPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 0.01, cmin=1)),
            'NAS100_USD': AssetConfig(name='NAS100_USD', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
            'US30_USD': AssetConfig(name='US30_USD', risk=0.01, sl=4.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
            'SPX500_USD': AssetConfig(name='SPX500_USD', risk=0.01, sl=4.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
            'USD_JPY': AssetConfig(name='USD_JPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 0.01, cmin=1)),
            #'USTEC_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
            'XAU_USD': AssetConfig(name='XAU_USD', risk=0.01, sl=4.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 0.01, cmin=1)), #
        }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        'trendContinuation':StrategyConfig(name='TC', assets={
            #'BTC_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 6.0, 'order_type':'stop', 'min_size':0.0001, 'max_size':10000000},
            #'ETH_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 6.0, 'order_type':'stop', 'min_size':0.0001, 'max_size':10000000},
            'GBP_JPY': AssetConfig(name='GBP_JPY', risk=0.01, sl=2.0, tp=6.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 0.01, cmin=1)),
            'NAS100_USD': AssetConfig(name='NAS100_USD', risk=0.01, sl=2.0, tp=6.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
            'SPX500_USD': AssetConfig(name='SPX500_USD', risk=0.01, sl=2.0, tp=6.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
            'USD_JPY': AssetConfig(name='USD_JPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=10000000, commission=Commissions('pershare', 0.01, cmin=1)),
            'XAG_USD': AssetConfig(name='XAG_USD', risk=0.01, sl=2.0, tp=6.0, order='stop', min_size=1, max_size=10000000, commission=Commissions('pershare', 1.0, cmin=1)),
            #'XBR_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 6.0, 'order_type':'stop', 'min_size':1000, 'max_size':10000000},
        }, use_sl=True, use_tp=True, time_limit=200, timeframe='H1'),
        'kamaTrend':StrategyConfig(name='TKAMA', assets={
            'NAS100_USD': AssetConfig(name='NAS100_USD', risk=0.01, sl=1.0, tp=4.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
            'US30_USD': AssetConfig(name='US30_USD', risk=0.01, sl=4.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=100, timeframe='H1'),
        'atrExt':StrategyConfig(name='ATRE', assets={
            'NAS100_USD': AssetConfig(name='NAS100_USD', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
            'SPX500_USD': AssetConfig(name='SPX500_USD', risk=0.01, sl=4.0, tp=8.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=10, timeframe='H1'),
        'turtlesBreakout':StrategyConfig(name='TBO', assets={
            'GBP_JPY': AssetConfig(name='GBP_JPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 0.01, cmin=1)),
            'NAS100_USD': AssetConfig(name='NAS100_USD', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
            'US30_USD': AssetConfig(name='US30_USD', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
            'SPX500_USD': AssetConfig(name='SPX500_USD', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=10000, commission=Commissions('pershare', 1.0, cmin=1)),
            'XAG_USD': AssetConfig(name='XAG_USD', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=10000000, commission=Commissions('pershare', 1.0, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
    }
    tickers = {'SP500': {'yfinance':'500.PA', 'degiro':'LU1681048804'}, 
               'NASDAQ': {'yfinance':'ANX.PA', 'degiro':'LU1681038243'}, 
               'STOXX': {'yfinance':'C50.PA'}, 
               'MSCI World': {'yfinance':'CW8.PA'}}
    strategies = {
        'detrended': StrategyConfig(name='DeTrend', assets={
            'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        'momentum': StrategyConfig(name='MOM', assets={
            'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        # 'modFisher': StrategyConfig(name='MFish', assets={
        #     'SP500': AssetConfig(name='SPY', risk=0.04, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        # }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        # 'fibEnvelopes': StrategyConfig(name='FiEnv', assets={
        #     'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=6.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        # }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        # 'slopeStrat': StrategyConfig(name='Slo', assets={
        #     'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        # }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        # 'robBookerStrat': StrategyConfig(name='Book', assets={
        #     'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=6.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        # }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        # 'mandiStrat': StrategyConfig(name='Mandi', assets={
        #     'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=6.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        # }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        'stochRsiStrat': StrategyConfig(name='SRSI', assets={
            'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        # 'maZone': StrategyConfig(name='MAZ', assets={
        #     'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        # }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        'paraSar': StrategyConfig(name='PARA', assets={
            'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        'trendInten': StrategyConfig(name='TI', assets={
            'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=350, timeframe='H1'),
        'rsiNeutrality': StrategyConfig(name='RSINeu', assets={
            'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        'catapultTrend': StrategyConfig(name='Cata', assets={
            'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        'turtlesBreakout': StrategyConfig(name='TB', assets={
            'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        'trendContinuation': StrategyConfig(name='TC', assets={
            'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=6.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        'kamaTrend': StrategyConfig(name='KAMA', assets={
            'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=10.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=350, timeframe='H1'),
        'atrExt': StrategyConfig(name='ATRE', assets={
           'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=4.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=50, timeframe='H1'),
        'indexPullback': StrategyConfig(name='IPB', assets={
            'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=10.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=350, timeframe='H1'),
    }

    strategies = {
        'dailyPB': StrategyConfig(name='DPB', assets={
            # 'LU1681038243': AssetConfig(name='QQQ', risk=0.02, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
            'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=2, timeframe='H1'),
        'volatPB': StrategyConfig(name='VPB', assets={
            'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=3, timeframe='H1'),
        'pullbackBounce': StrategyConfig(name='Volat', assets={
            'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=2, timeframe='H1'),
        # 'indexPullback': StrategyConfig(name='IPB', assets={
        #     'SP500': AssetConfig(name='SPY', risk=0.015, sl=4.0, tp=10.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        # }, use_sl=True, use_tp=True, time_limit=10, timeframe='H1'),
        'atrExt': StrategyConfig(name='ATRE', assets={
           'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=10.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=False, time_limit=10, timeframe='H1'),
        'kamaTrend': StrategyConfig(name='KAMA', assets={
            'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=10.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=10, timeframe='H1'),
        'rsiNeutrality': StrategyConfig(name='RSINeu', assets={
            'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=10, timeframe='H1'),
        'paraSar': StrategyConfig(name='PARA', assets={
            'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=5.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=5, timeframe='H1'),
        'momentum': StrategyConfig(name='MOM', assets={
            'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=5.0, order='stop', min_size=1, max_size=5000, commission=Commissions('perunit', 0.05, cmin=1)),
        }, use_sl=True, use_tp=True, time_limit=5, timeframe='H1'),
    }
    

    broker = 'degiro'
    if broker == 'degiro':
        dg = DeGiro('OneMade','Onemade3680')

    # Prepare data needed for backtest
    signals = Signals(backtest=True, side=Signals.Side.LONG, errors=False)
    indicators = Indicators(errors=False)
    total_data = []
    data = {}
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

    total_data.append(df)

    # Backtest
    bt = BackTest(strategies=strategies, 
                config=config)
    
    total_data = pd.concat(total_data)
    trades = bt.backtest(df=total_data)

    bt.btPlot(balance=True, log=True)
    stats = bt.stats(column='StratName')

    complete = trades[trades['OneCandle'] == False].copy()
    final_df = pd.DataFrame()
    final_df['OrderDate'] = complete['OrderTime']
    final_df['EntryDate'] = complete['EntryTime']
    final_df['ID'] = [''] * len(final_df)
    final_df['Strategy'] = complete['Strategy']
    final_df['Ticker'] = complete['Ticker']
    final_df['Side'] = np.where(complete['Signal'] == 'long', 'buy', 'sell')
    final_df['Entry'] = complete['Entry']
    final_df['Slippage'] = [0]*len(final_df)
    final_df['Spread'] = complete['Spread']
    final_df['Commission'] = complete['Commission']
    final_df['Locate'] = [0]*len(final_df)
    final_df['SL'] = complete['SL']
    final_df['TP'] = complete['TP']
    final_df['OrderType'] = complete['Order'].str.upper()
    final_df['Executed'] = [True] * len(final_df)
    final_df['ExitDate'] = complete['ExitTime']
    final_df['Exit'] = complete['Exit']
    final_df['Realized'] = (final_df['Exit'] - final_df['Entry'])/(final_df['Entry'] - final_df['SL'])
    
    final_df.reset_index(drop=True, inplace=True)
    temp = []
    for i in final_df.index:
        x = final_df.loc[i]
        temp.append(f"{dt.datetime.timestamp(x['OrderDate'])}_{x['Entry']}")
    final_df['ID'] = temp

    final_df.to_excel('backtest.xlsx')
