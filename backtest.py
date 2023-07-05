
import os
import datetime as dt
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signals import Signals
from indicators import OHLC, Indicators
from google_sheets.google_sheets import GoogleSheets


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
        self.order = order
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
            'order_type': self.order, 
            'min_size': self.min_size, 
            'max_size': self.max_size
        }

class StrategyConfig:

    def __init__(self, name:str, assets:dict={}, use_sl:bool=True, use_tp:bool=True, 
                 time_limit:int=50, timeframe:str='H1') -> None:

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
        }
    
class BtConfig:

    '''
    Class used to create the backtest config.
    '''

    def __init__(self, init_date:str, final_date:str, capital:float=10000.0,
                use_sl:bool=True, use_tp:bool=True, time_limit:int=365,
                min_size:float=1, max_size:float=10000000, commission:Commissions=None,
                max_trades:int=3, filter_ticker:bool=True, filter_strat:bool=False,
                reset_orders:bool=True, continue_onecandle=True
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
        '''

        self.capital = capital
        self.init_date = dt.datetime.strptime(init_date, '%Y-%m-%d') \
                        if isinstance(init_date, str) else init_date
        self.final_date = dt.datetime.strptime(final_date, '%Y-%m-%d') \
                        if isinstance(final_date, str) else final_date
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
            'init_date': self.init_date,
            'final_date': self.final_date,
            'use_sl': self.use_sl,
            'use_tp': self.use_tp,
            'time_limit': self.time_limit,
            'min_size': self.min_size,
            'max_size': self.max_size,
            'commission': self.commission,
        }

class Trade:

    def __init__(self, candle:dict, signal:str, strategy:StrategyConfig,
                    entry:float, slprice:float, tpprice:float, 
                    size:float, balance:float, asset:AssetConfig) -> None:

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
        order: str
            Order type. It can be 'market', 'limit' or 'stop'.
        entry: float
            Entry price.
        slprice: float
            SL price.
        tpprice: float
            TP price.
        commission: Commissions
            Commissions object associated to the trade.
        size: float
            Size of the trade.
        risk: float
            Risk of the trade.
        balance: float
            Balance when entering the trade.
        '''

        # comission = asset.commission.value * 100 if 'JPY' in candle['Ticker'] and \
        #             asset.commission.type == 'pershare' else asset.commission.value
        strategy.assets = [t for t in strategy.assets.keys()]

        self.datetime = candle['DateTime']
        self.entrytime =  candle['DateTime']
        self.ticker = candle['Ticker']
        self.strategy = strategy
        self.order = asset.order
        self.signal = signal
        self.entry = entry
        self.exit = candle['Close']
        self.sl = slprice
        self.tp = tpprice
        self.returns = candle['Close'] - entry
        self.spread = candle['Spread']
        self.commission = asset.commission
        self.risk = asset.risk
        self.balance = balance
        self.size = size if balance > 0 else 0.0
        self.sldist = abs(entry - slprice)
        self.high = candle['High']
        self.low = candle['Low']
        self.candles = [candle]
        self.onecandle = False
        self.asset = asset
    
    def calculateCommission(self):

        '''
        Calculates the commission applied to the trade.
        '''

        commission = self.commission.value * self.size if self.commission.type == 'pershare' \
                    else self.commission.value * self.returns
        
        if commission > self.commission.max:
            commission = self.commission.max
        elif commission < self.commission.min:
            commission = self.commission.min

        self.commission = commission

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
            'Ticker': self.ticker,
            'Strategy': self.strategy.to_dict(),
            'Order': self.order,
            'Signal': self.signal,
            'Entry': self.entry,
            'Exit': self.exit,
            'SL': self.sl,
            'TP': self.tp,
            'Return': self.returns,
            'Spread': self.spread,
            'Comsission': self.calculateCommission(),
            'CommissionStruc': self.commission.to_dict(),
            'Risk': self.risk,
            'Balance': self.balance,
            'Size': self.size if self.balance > 0 else 0.0,
            'SLdist': abs(self.entry - self.sl),
            'High': max([c['High'] for c in self.candles]),
            'Low': min([c['Low'] for c in self.candles]),
            'Candles': self.candles,
            'OneCandle': self.onecandle,
            'Asset': self.asset.to_dict()
        }

class BackTest(OHLC):

    '''
    Class used to carry out the backtest of the strategy.
    '''
    
    config = BtConfig('2018-01-01', dt.date.today().strftime('%Y-%m-%d'), capital=10000.0, 
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

    def fillHalts(self, df:pd.DataFrame):

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
    
    def getEntries(self, candle):
        
        self.entries = [c.replace('Entry','').replace('entry','') for c in candle.columns \
                        if 'entry' in c.lower()]
    
    def getExits(self, candle):
        
        self.exits = [c.replace('Exit','').replace('exit','') for c in candle.columns \
                        if 'exit' in c.lower()]

    def openQty(self, candle, open_trades:list, filter_ticker:bool=True, 
                filter_strat:bool=False):

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

    def backtest(self, df:pd.DataFrame=None) -> pd.DataFrame():

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
        self.data_df = self._newDf(df, needed_cols=['Open', 'High', 'Low', 'Close', 'Spread', 'SLdist', 'TPdist'], 
                                   overwrite=True)

        # Initialize variables
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

            # Iterate for each asset in this DateTime
            for i in g[1].index:

                candle = g[1].loc[i]

                if candle['SLdist'] != candle['SLdist']:
                    continue
                if candle['TPdist'] != candle['TPdist']:
                    continue
                
                # Check if we are between the backtest dates
                if candle['Date'] < self.config.init_date or candle['Date'] > self.config.final_date:
                    continue
                    
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
                        if candle[strat] != 0 and trades_qty < self.config.max_trades:

                            asset = self.strategies[strat].assets[candle['Ticker']]
                            risk = asset.risk
                            sldist = candle['SLdist']
                            tpdist = candle['TPdist']
                            size = risk/sldist*balance[-1]
                            if size > asset.max_size:
                                size = asset.max_size
                            elif size < asset.min_size:
                                size = asset.min_size

                            # Long orders
                            if candle[strat] > 0:
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
                                if not entered:
                                    trade = Trade(candle, 'long', self.strategies[strat], entry, entry - sldist, 
                                                  entry + tpdist, size, balance[-1], asset)

                                    # If market order execute it if not append to orders
                                    if asset.order_type == 'market':
                                        open_trades.append(trade)
                                    else:
                                        open_orders.append(trade)

                            # Short orders
                            if candle[strat] < 0:
                                # Sell order entry price
                                trade_order = asset.order_type
                                if asset.order_type == 'market':
                                    entry = candle['Open'] - candle['Spread']
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
                                if not entered:
                                    trade = Trade(candle, 'short', self.strategies[strat], entry, entry + sldist, 
                                                  entry - tpdist, size, balance[-1], asset)

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
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- Aqui te has quedadoooo
                            # LIMIT orders
                            elif order.order == 'limit':

                                if order.signal == 'short':

                                    if order.entry < candle['High']:
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

                                if order.signal == 'long':

                                    if order.entry > candle['Low'] + order.spread:
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
                    
                    # Delete from open orders if already executed
                    for d in delete:
                        open_orders.remove(d)

                # Store trade evolution
                for trade in open_trades:

                    if trade['Ticker'] == candle['Ticker']:
                        trade['Candles'].append({'DateTime':candle['DateTime'] ,'Open':candle['Open'], 'High':candle['High'], 
                                        'Low':candle['Low'], 'Close':candle['Close'], 'Volume': candle['Volume']})

                # Check open trades limits orders
                if len(open_trades) > 0:
                    
                    delete = []
                    for trade in open_trades:

                        if candle['Ticker'] == trade['Ticker']:
                            
                            exited = False

                            # Check SL
                            if self.config['use_sl']:

                                if trade['Order'] != 'stop' or not continue_onecandle or len(trade['Candles']) > 1:

                                    if trade['Signal'] == 'Sell' and candle['High'] + trade['Spread'] >= trade['SL']: # High
                                        trade['Exit'] = trade['SL'] if candle['Open'] < trade['SL'] else candle['Open']
                                        trade['Return'] = trade['Entry'] - trade['Exit']
                                        exited = True
                                        if len(trade['Candles']) <= 1 and candle['Low'] < trade['TP'] and candle['High'] > trade['SL']:
                                            trade['OneCandle'] = True

                                    if trade['Signal'] == 'Buy' and candle['Low'] <= trade['SL']: # Low
                                        trade['Exit'] = trade['SL'] if candle['Open'] > trade['SL'] else candle['Open']
                                        trade['Return'] = trade['Exit'] - trade['Entry']
                                        exited = True
                                        if len(trade['Candles']) <= 1 and candle['High'] > trade['TP'] and candle['Low'] < trade['SL']:
                                            trade['OneCandle'] = True
                            
                            # Check TP
                            if self.config['use_tp'] and not exited:

                                if trade['Order'] != 'limit' or not continue_onecandle or len(trade['Candles']) > 1:

                                    if trade['Signal'] == 'Sell' and candle['Low'] + trade['Spread'] <= trade['TP']: #Low
                                        trade['Exit'] = trade['TP'] if candle['Open'] > trade['TP'] else candle['Open']
                                        trade['Return'] = trade['Entry'] - trade['Exit']
                                        exited = True
                                        if len(trade['Candles']) <= 1 and candle['Low'] < trade['TP'] and candle['High'] > trade['SL']:
                                            trade['OneCandle'] = True

                                    if trade['Signal'] == 'Buy' and candle['High'] >= trade['TP']: #High
                                        trade['Exit'] = trade['TP'] if candle['Open'] < trade['TP'] else candle['Open']
                                        trade['Return'] = trade['Exit'] - trade['Entry']
                                        exited = True
                                        if len(trade['Candles']) <= 1 and candle['High'] > trade['TP'] and candle['Low'] < trade['SL']:
                                            trade['OneCandle'] = True
                            
                            # Check time limit
                            if not exited and self.config['time_limit'] > 0 and len(trade['Candles']) >= self.config['time_limit']:
                                trade['Exit'] = candle['Close']
                                if trade['Signal'] == 'Buy':
                                    trade['Return'] = trade['Exit'] - trade['Entry']
                                elif trade['Signal'] == 'Sell':
                                    trade['Return'] = trade['Entry'] - trade['Exit']
                                exited = True

                            if exited:
                                trade['Result'] = trade['Return'] * trade['Size'] - trade['Comission']
                                trade['ExitTime'] = candle['DateTime']
                                closed_trades.append(trade)
                                date_result += trade['Result']
                                delete.append(trade)
                    
                    # Delete open trades if already exited
                    for d in delete:
                        open_trades.remove(d)
                
                # Check open trades Exits if the strategy has exit conditions
                if 'Exit' in self.data_df.columns and candle['Exit'] != 0:

                    delete = []
                    for trade in open_trades:
                        if trade['Ticker'] == candle['Ticker']:

                            exited = False

                            # Exit Buy
                            if candle['Exit'] == 1 and trade['Signal'] == 'Buy':
                                trade['Exit'] = candle['Open']
                                trade['Return'] = trade['Exit'] - trade['Entry']
                                exited = True
                            # Exit Sell
                            elif candle['Exit'] == -1 and trade['Signal'] == 'Sell':
                                trade['Exit'] = candle['Open'] + trade['Spread']
                                trade['Return'] = trade['Entry'] - trade['Exit']
                                exited = True
                                    
                            if exited:
                                trade['Result'] = trade['Return'] * trade['Size'] - trade['Comission']
                                trade['ExitTime'] = candle['DateTime']
                                closed_trades.append(trade)
                                date_result += trade['Result']
                                delete.append(trade)
                    
                    for d in delete:
                        open_trades.remove(d)

                prev_candle[candle['Ticker']] = candle

            balance.append(balance[-1]+date_result)

        # Calculate and store final data
        trades = pd.DataFrame(closed_trades)
        
        if not trades.empty:
            trades.loc[:,'%ret'] = trades['Result'] / trades['Balance']
            trades.loc[:,'BalanceExit'] = trades['Balance'] + trades['Result'] #trades['Balance'].tolist()[1:] + [balance[-1]]
            trades.loc[:,'AccountPeak'] = trades['Balance'].cummax()
            trades.loc[:,'AccountDD'] = 1 - trades['Balance']/trades['AccountPeak']

        self.trades = trades
        self.open_trades = pd.DataFrame(open_trades)
        self.open_orders = pd.DataFrame(open_orders)
        self.closed_trades = closed_trades

        return self.trades

    def saveResult(self, file:str='TradesBacktested,xlsx', sheet:str='CompleteTrades'):

        writer = pd.ExcelWriter(file)
        self.trades.to_excel(writer, sheet_name=sheet, index=False)
        writer.save()

    def resultsToGoogle(self, sheetid:str, sheetrange:str):

        google = GoogleSheets(sheetid, secret_path=os.path.join('google_sheets','client_secrets.json'))
        google.appendValues(self.trades.values, sheetrange=sheetrange)

    def btKPI(self, print_stats=True):
        
        self.trades['Ret'] = self.trades['Result']/(abs(self.trades['Entry']-self.trades['SL'])*self.trades['Size']) * self.trades['Risk']
        self.trades['CumRet'] = (1+self.trades['Ret']).cumprod()

        # Backtest analysis
        winrate = len(self.trades['Return'][self.trades['Return'] > 0.0])/len(self.trades['Return'])
        avg_win = self.trades['%ret'][self.trades['%ret'] > 0].mean()
        avg_loss = self.trades['%ret'][self.trades['%ret'] < 0].mean()
        expectancy = (winrate*avg_win - (1-winrate)*abs(avg_loss))

        days = np.busday_count(self.data_df['Date'].tolist()[0].date(), self.data_df['Date'].tolist()[-1].date())

        if print_stats:
            print(f'Winrate: {winrate :%}') # With two decimal spaces and commas for the thousands :,.2f
            print(f'Avg. Win: {avg_win :%}')
            print(f'Avg. Loss: {avg_loss :%}')
            print(f'Expectancy: {expectancy :%}')
            print(f'Trading frequency: {len(self.trades)/days * 100//1/100}')
            #print(f'Monthly Expectancy: {((1 + expectancy*len(trades)/days)**(20) - 1) :%}')
            #print(f'Anual Expectancy: {((1 + expectancy*len(trades)/days)**(52*5) - 1) :%}')
            print(f'Kelly: {(winrate*avg_win - (1-winrate)*abs(avg_loss))/avg_win :%}')
            print(f"Backtest Max. DD: {self.trades['AccountDD'].max() :%}")
            # print(f'Ulcer Index: {(((trades['AccountDD'] * 100)**2).sum()/len(trades['AccountDD']))**(1/2) * 100//1/100}')
            
        stats = {
            'Winrate': winrate,
            'AvgWin': avg_win,
            'AvgLoss': avg_loss,
            'Expectancy': expectancy,
            'Days': days,
            'Frequency': len(self.trades)/days * 100//1/100,
            'Kelly': (winrate*avg_win - (1-winrate)*abs(avg_loss))/avg_win,
            'MaxDD': self.trades['AccountDD'].max()
        }

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

    def btPlot(self, log:bool=True):

        # Plot Backtest results
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.0,
                            row_heights=[3,1],
                            specs=[[{'secondary_y': True}],[{'secondary_y': False}]])

        fig.add_trace(go.Scatter(x=self.trades['ExitTime'],y=self.trades['BalanceExit'], name='Balance'), row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=self.trades['ExitTime'], y=self.trades['AccountDD'] * 10000//1/100, fill='tozeroy', name='DrawDown'), row=1, col=1, secondary_y=True)

        fig.update_yaxes(title_text='Return ($)', row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text='DrawDown (%)', row=1, col=1, secondary_y=True)

        if log:
            fig.update_yaxes(type="log", row=1, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(x=self.trades['ExitTime'][self.trades['%ret'] > 0.0],y=self.trades['%ret'][self.trades['%ret'] > 0.0] * 10000//1/100, 
                                name='Wins', marker_color='green', mode='markers'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.trades['ExitTime'][self.trades['%ret'] <= 0.0], y=self.trades['%ret'][self.trades['%ret'] <= 0.0] * 10000//1/100, 
                                name='Losses', marker_color='red', mode='markers'), row=2, col=1)

        fig.update_yaxes(title_text='Return (%)', row=2, col=1)
            
        # fig.add_trace(go.Histogram(x=trades['%ret'][trades['%ret'] > 0], name='Wins', marker_color='green'), row=3, col=1)
        # fig.add_trace(go.Histogram(x=trades['%ret'][trades['%ret'] <= 0], name='Losses', marker_color='red'), row=3, col=1)

        # if log:
        #     fig.update_yaxes(type="log", row=3, col=1)

        # fig.update_yaxes(title_text='Qty.', row=3, col=1)
        # fig.update_xaxes(title_text='Return (%)', row=3, col=1)

        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_layout(title=f"Account balance {self.config['capital'] + self.trades['Result'].sum()*100//1/100}$", autosize=False,
                            width=1000,
                            height=700)

        fig.show()
    
    def tickerStats(self, plot:bool=False, print_stats:bool=False, print_comparison:bool=True):

        pair_stats = []
        pairs = []
        for g in self.trades.groupby('Ticker'):

            temp = g[1]
            temp.reset_index(drop=True, inplace=True)
            ret = temp['Return'].tolist()
            risk = temp['Risk'].tolist()
            entry = temp['Entry'].tolist()
            balance = [self.config['capital']]
            result = []
            pct_ret = []
            size = []
            for i in range(len(temp)):
                size.append(balance[-1]*risk[i]/(temp['SLdist'].tolist()[i]))
                result.append(size[-1]*(ret[i] - self.config['commission']))
                pct_ret.append(result[-1]/balance[-1])
                balance.append(balance[-1]+result[-1])

            temp.loc[:,'Size'] = size
            temp.loc[:,'Balance'] = balance[1:]
            temp.loc[:,'Result'] = result
            temp.loc[:,'%ret'] = pct_ret
            temp.loc[:,'AccountPeak'] = temp['Balance'].cummax()
            temp.loc[:,'AccountDD'] = 1 - temp['Balance']/temp['AccountPeak']

            winrate = len(temp[temp['Result'] > 0])/len(temp)
            avg_win = temp['%ret'][temp['%ret'] > 0].mean()
            avg_loss = temp['%ret'][temp['%ret'] < 0].mean()

            if print_stats:
                print(f'{g[0]} ---------------------------------------------')
                print('')
                print('Winrate: ',winrate * 10000//1/100,'%')
                print('Avg. Win: ',avg_win * 10000//1/100,'%')
                print('Avg. Loss: ',avg_loss * 10000//1/100,'%')
                print('Expectancy: ',(winrate*avg_win - (1-winrate)*abs(avg_loss)) * 10000//1/100,'%')
                print('Kelly: ',(winrate*avg_win - (1-winrate)*abs(avg_loss))/avg_win * 10000//1/100,'%')
                print('Max. DD: ', temp['AccountDD'].max() * 10000//1/100, '%')

            if plot:
                fig = make_subplots(specs=[[{'secondary_y': True}]])
                fig.add_trace(go.Scatter(x=temp['Date'],y=temp['Balance'], name='Balance'), secondary_y=False)
                fig.add_trace(go.Scatter(x=temp['Date'], y=temp['AccountDD'] * 10000//1/100, fill='tozeroy', name='DrawDown'), secondary_y=True)

                fig.update_xaxes(title_text='Date')
                fig.update_yaxes(title_text='Return ($)', secondary_y=False)
                fig.update_yaxes(title_text='DrawDown (%)', secondary_y=True)
                fig.update_layout(title=f"Account balance {self.config['capital'] + temp['Result'].sum()*100//1/100}$", 
                                autosize=False,width=1000,height=700,)
                fig.show()

            pair_stats.append({
                'Ticker': g[0],
                'Winrate': winrate,
                'Avg. Win': avg_win,
                'Avg. Loss': avg_loss,
                'Expectancy': (winrate*avg_win - (1-winrate)*abs(avg_loss)),
                'Kelly': (winrate*avg_win - (1-winrate)*abs(avg_loss))/avg_win,
                'Avg risk': temp['Risk'].mean(),
                'BtBalance': temp['Balance'].tolist()[-1],
                'MaxDD': temp['AccountDD'].max(),
                '#trades': len(temp)
            })
            pairs.append(g[0])
        
        for strat in self.pairs_config:
            for pair in self.pairs_config[strat]:
                if pair not in pairs:
                    pairs.append(pair)
                    pair_stats.append({
                        'Ticker': pair,
                        'Winrate': float('nan'),
                        'Avg. Win': 0.0,
                        'Avg. Loss': 0.0,
                        'Expectancy': 0.0,
                        'Kelly': 0.0,
                        'Avg risk': 0.0,
                        'BtBalance': self.config['capital'],
                        'MaxDD': 0.0
                    })

        # Comparison of pairs and total
        stats_df = pd.DataFrame(pair_stats)
        stats_df.sort_values(by=['Kelly'], ascending=False, inplace=True)
        stats_df.reset_index(drop=True, inplace=True)
        if print_comparison:
            print(stats_df)

        return stats_df

    def weekdayStats(self, plot:bool=False, print_stats:bool=True):
        
        self.trades['WeekDay'] = self.trades['EntryTime'].dt.day_name()
        day_stats = []
        for day in ['Monday','Tuesday','Wednesday','Thursday','Friday']:

            temp = self.trades.groupby('WeekDay').get_group(day)
            temp.reset_index(drop=True, inplace=True)
            ret = temp['Return'].tolist()
            risk = temp['Risk'].tolist()
            entry = temp['Entry'].tolist()
            balance = [self.config['capital']]
            result = []
            pct_ret = []
            size = []
            for i in range(len(temp)):
                size.append(balance[-1]*risk[i]/(temp['SLdist'].tolist()[i]))
                result.append(size[-1]*(ret[i] - self.config['commission']))
                pct_ret.append(result[-1]/balance[-1])
                balance.append(balance[-1]+result[-1])

            temp.loc[:,'Size'] = size
            temp.loc[:,'Balance'] = balance[:-1]
            temp.loc[:,'Result'] = result
            temp.loc[:,'%ret'] = pct_ret
            temp.loc[:,'AccountPeak'] = temp['Balance'].cummax()
            temp.loc[:,'AccountDD'] = 1 - temp['Balance']/temp['AccountPeak']

            winrate = len(temp[temp['Result'] > 0])/len(temp)
            avg_win = temp['%ret'][temp['%ret'] > 0].mean()
            avg_loss = temp['%ret'][temp['%ret'] < 0].mean()

            if print_stats:
                print(f'{day} ---------------------------------------------')
                print('')
                print('Winrate: ',winrate * 10000//1/100,'%')
                print('Avg. Win: ',avg_win * 10000//1/100,'%')
                print('Avg. Loss: ',avg_loss * 10000//1/100,'%')
                print('Expectancy: ',(winrate*avg_win - (1-winrate)*abs(avg_loss)) * 10000//1/100,'%')
                print('Kelly: ',(winrate*avg_win - (1-winrate)*abs(avg_loss))/avg_win * 10000//1/100,'%')
                print('Max. DD: ', temp['AccountDD'].max() * 10000//1/100, '%')

            if plot:
                fig = make_subplots(specs=[[{'secondary_y': True}]])
                fig.add_trace(go.Scatter(x=temp['Date'],y=temp['Balance'], name='Balance'), secondary_y=False)
                fig.add_trace(go.Scatter(x=temp['Date'], y=temp['AccountDD'] * 10000//1/100, fill='tozeroy', name='DrawDown'), secondary_y=True)

                fig.update_xaxes(title_text='Date')
                fig.update_yaxes(title_text='Return ($)', secondary_y=False)
                fig.update_yaxes(title_text='DrawDown (%)', secondary_y=True)
                fig.update_layout(title=f"Account balance {self.config['capital'] + temp['Result'].sum()*100//1/100}$", 
                                autosize=False,width=1000,height=700,)
                fig.show()

            day_stats.append({
                'Weekday': day,
                'Winrate': winrate,
                'Avg. Win': avg_win,
                'Avg. Loss': avg_loss,
                'Expectancy': (winrate*avg_win - (1-winrate)*abs(avg_loss)),
                'Kelly': (winrate*avg_win - (1-winrate)*abs(avg_loss))/avg_win,
                'Avg risk': temp['Risk'].mean(),
                'BtBalance':temp['Balance'].tolist()[-1],
                'MaxDD':temp['AccountDD'].max()
            })



        # Comparison of pairs and total
        stats_df = pd.DataFrame(day_stats)
        stats_df.sort_values(by=['Kelly'], ascending=False, inplace=True)
        stats_df.reset_index(drop=True, inplace=True)
        if print_stats:
            print(stats_df)

        return stats_df



if __name__ == '__main__':

    import yfinance as yf

    config = {
        'capital': 10000.0,
        'init_date': '2010-01-01',
        'final_date': '2023-12-31',
        'lots': False,
        'commission_type': 'pershare', # 'pershare', 'percentage'
        'commission': 0.000035,
    }

    config = BtConfig('2018-01-01', (dt.date.today() - dt.timedelta(days=250)).strftime('%Y-%m-%d'), capital=10000.0, 
                      use_sl=True, use_tp=True, time_limit=None, min_size=1000, 
                      max_size=10000000, commission=Commissions())


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



    # Prepare data needed for backtest
    signals = Signals(backtest=True)
    indicators = Indicators()
    data = {}
    complete = []
    for strat in strategies:

        data[strat] = {}
        if strat not in dir(signals):
            continue
        signal = getattr(signals, strat)

        for t in strategies[strat]['instruments']:
            temp = yf.Ticker('SPY').history(period='2y',interval='1h')
            temp.loc[:,'SLdist'] = strategies[strat]['instruments'][t]['SL'] * indicators.atr(n=20, method='s', dataname='ATR', new_df=temp)['ATR'] \
                                    if strategies[strat]['instruments'][t]['volatility'] else strategies[strat]['instruments'][t]['SL']
            temp.loc[:,'TPdist'] = strategies[strat]['instruments'][t]['TP'] * indicators.atr(n=20, method='s', dataname='ATR', new_df=temp)['ATR'] \
                                    if strategies[strat]['instruments'][t]['volatility'] else strategies[strat]['instruments'][t]['TP']
            temp.loc[:,'Ticker'] = [t]*len(temp)
            temp.loc[:,'Date'] = pd.to_datetime(temp.index)
            
            if 'Volume' not in temp.columns:
                temp['Volume'] = [0]*len(temp)

            temp = signal(df=temp, strat_name=strat)
            temp['Entry'] = temp[strat].apply(lambda x: [{'strategy':strategies[strat].name, 'side':x}])
            data[strat][t] = temp.copy()


        # Backtest
        bt = BackTest(pairs_config={strat: strategies[strat]['instruments']}, 
                    config={**config, **{k: strategies[strat][k] for k in strategies[strat] \
                                        if k not in ['name','instruments','timeframe']}})

        # Show main metrics
        df = pd.concat([data[strat][t] for t in data[strat]], ignore_index=True)
        df.sort_values('Date', inplace=True)
        df.loc[:,'DateTime'] = pd.to_datetime(df['DateTime'], unit='s')
        df['Close'] = df['Close'].fillna(method='ffill')
        df['Spread'] = df['Spread'].fillna(method='ffill')
        df['Open'] = np.where(df['Open'] != df['Open'], df['Close'], df['Open'])
        df['High'] = np.where(df['High'] != df['High'], df['Close'], df['High'])
        df['Low'] = np.where(df['Low'] != df['Low'], df['Close'], df['Low'])

        trades = bt.backtest(df=df, reset_orders=True)

        # bt.btPlot(log=True)

        complete.append(trades[trades['OneCandle'] == False])

    complete = pd.concat(complete)
    final_df = pd.DataFrame()
    final_df['OrderDate'] = complete['OrderTime']
    final_df['EntryDate'] = complete['EntryTime']
    final_df['ID'] = [''] * len(final_df)
    final_df['Strategy'] = complete['Strategy']
    final_df['Ticker'] = complete['Ticker']
    final_df['Side'] = np.where(complete['Signal'] == 'Buy', 'buy', 'sell')
    final_df['Entry'] = complete['Entry']
    final_df['Slippage'] = [0]*len(final_df)
    final_df['Spread'] = complete['Spread']
    final_df['Commission'] = [0]*len(final_df)
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
