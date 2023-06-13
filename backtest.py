
import os
import datetime as dt
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from signals import Signals
from oanda import Oanda
from indicators import Indicators
from google_sheets.google_sheets import GoogleSheets

class BackTest:

    '''
    Class used to carry out the backtest of the strategy.
    '''

    config = {
        'capital': 10000.0,
        'init_date': '2018-01-01',
        'final_date': '2022-09-27',
        'lots': True,
        'min_size': 1000,
        'max_size': 10000000,
        'commission': 0.000035,
    }

    def __init__(self, pairs_config:dict, config:dict=None) -> None:

        '''
        Starts the Backtesting object.

        Parameters
        ----------
        pairs_config: dict
            Dictionary with the data for the pairs.
            Example:
            ex_dict = {
                'strategy1': {
                    'PAIR1':{'commission':0.02,'risk':0.01},
                    'PAIR2':{'commission':0.05,'risk':0.015},
                    ...
                },
                ...
            }
        config: dict
            Dictionary with the data for the strategy.
            Example:
            ex_dict = {
                'capital': 1000.0,
                'init_date': '2000-01-01',
                'final_date': '2020-11-01',
                'use_sl': True,
                'use_tp': True,
                'time_limit': 5,
                'lots': True,
                'min_size': 1000,
                'max_size': 10000000,
                'commission': 0.000035,
            }
        '''

        self.pairs_config = pairs_config
        self.config = config if config != None else self.config
        if isinstance(self.config['init_date'], str):
            self.config['init_date'] = dt.datetime.strptime(self.config['init_date'], '%Y-%m-%d')
        if isinstance(self.config['final_date'], str):
            self.config['final_date'] = dt.datetime.strptime(self.config['final_date'], '%Y-%m-%d')

    def fillNan(self, df:pd.DataFrame):

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
    
    def createTrade(self, candle:dict, signal:str, strategy:str, order:str, 
                    entry:float, slprice:float, tpprice:float, comision:float, 
                    size:float, risk:float, balance:float) -> dict:

        return {
            'OrderTime': candle['DateTime'],
            'EntryTime': candle['DateTime'],
            'Ticker': candle['Ticker'],
            'Strategy': strategy,
            'Order': order,
            'Signal': signal,
            'Entry': entry,
            'Exit': candle['Close'],
            'SL': slprice,
            'TP': tpprice,
            'Return': candle['Close'] - entry,
            'Spread': candle['Spread'],
            'Comission': comision * size,
            'Risk': risk,
            'Balance':balance,
            'Size': size if balance > 0 else 0.0,
            'SLdist': abs(entry - slprice),
            'High': candle['High'],
            'Low': candle['Low'],
            'Candles': [],
            'OneCandle': False,
        }

    def dataPrepare(self, ohlc:dict, unit:str='ms'):

        '''
        Formats the data to store all the pairs together with the correct 
        DateTime format and ordered by Date.

        Parameters
        ----------
        ohlc: dict
            Dictionary containing the candles DataFrames for all the pairs.
            Structure:
            example_dict = {
                'PAIR1': pd.DataFrame(),
                'PAIR2': pd.DataFrame(),
                ...
            }

        Returns
        -------
        data_df: pd.DataFrame
            Contains the candles of all the pairs orderes by Date with 
            the correct format.
        '''

        dfs = []
        for t in ohlc:
            dfs.append(ohlc[t])
            
        self.data_df = pd.concat(dfs, ignore_index=True)
        self.data_df.sort_values('Date', inplace=True)
        self.data_df.loc[:,'DateTime'] = pd.to_datetime(self.data_df['DateTime'], unit=unit)
        self.data_df = self.data_df[self.data_df['Entry'] == self.data_df['Entry']]
    
        return self.data_df

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
        # Filtered by ticker
        if filter_ticker:
            # Store the values for all the open trades
            for trade in open_trades:
                # If new ticker
                if trade['Ticker'] not in qty:
                    qty[trade['Ticker']] = {trade['Strategy']: 1} if filter_strat else 1
                # For already added tickers
                else:
                    if filter_strat:
                        if trade['Strategy'] not in qty[trade['Ticker']]:
                            qty[trade['Ticker']] = {trade['Strategy']: 1}
                        else:
                            qty[trade['Ticker']][trade['Strategy']] += 1
                    else:
                        qty[trade['Ticker']] += 1
            # If the current iteration ticker has no open trades add it
            if candle['Ticker'] not in qty:
                if filter_strat:
                    qty[candle['Ticker']] = {}
                    for strat in candle['Entry']:
                        qty[candle['Ticker']][strat['strategy']] = 0
                else:
                    qty[candle['Ticker']] = 0
            if filter_strat:
                for strat in candle['Entry']:
                    if strat['strategy'] not in qty[candle['Ticker']]:
                        qty[candle['Ticker']][strat['strategy']] = 0

        # Filtered by strat
        elif filter_strat and not filter_ticker:
            # Store the values for all the open trades
            for trade in open_trades:
                if trade['Strategy'] not in qty:
                    qty[trade['Strategy']] = 1
                else:
                    qty[trade['Strategy']] += 1
            # If the current iteration strategies have no open trades add them
            for strat in candle['Entry']:
                if strat['strategy'] not in qty:
                    qty[strat['strategy']] = 0
        # Not filtered
        else:
            qty = len(open_trades)
        
        return qty

    def backtest(self, df:pd.DataFrame=None, 
                 max_trades:int=3, filter_ticker:bool=True, filter_strat:bool=False,
                 reset_orders:bool=True, continue_onecandle=True) -> pd.DataFrame():

        '''
        Carries out the backtest logic.

        Parameters
        ----------
        df: pd.DataFrame
            Contains the complete candle data.
        max_trades: int
            Maximum open trades in any moment.
        filter_ticker: bool
            True to filter open trades for each symbol.
        filter_strat: bool
            True to filter open trades for each strategy.

        Returns
        -------
        trades: pd.DataFrame
            Contains the trades carried out during the backtest.
        '''

        # Check if the needed data is in the dataframe
        if df != None:
            self.data_df = df
        if 'SLdist' not in self.data_df:
            raise('ERROR: SLdist column not in DataFrame.')
        if 'Entry' not in self.data_df:
            raise('ERROR: Entry column not in DataFrame.')

        # Initialize variables
        open_trades = []
        open_orders = []
        closed_trades = []
        prev_candle = {}
        for t in self.pairs_config:
            prev_candle[t] = []
        balance = [self.config['capital']]

        # Group data by DateTime
        for g in self.data_df.groupby('DateTime'):
          
            date_result = 0
            #print('Ordenes: ',len(open_orders), 'Trades: ',len(open_trades))

            # Iterate for each candle in this DateTime
            for i in g[1].index:

                candle = g[1].loc[i]

                if candle['SLdist'] != candle['SLdist']:
                    continue
                if candle['TPdist'] != candle['TPdist']:
                    continue
                
                # Check if we are between the backtest dates
                if candle['Date'] < self.config['init_date'] or candle['Date'] > self.config['final_date']:
                    continue
                    
                # Look for entries
                if len(candle['Entry']) > 0:

                    trade = {}

                    for strat in candle['Entry']:
                      
                        risk = self.pairs_config[strat['strategy']][candle['Ticker']]['risk']
                        sldist = candle['SLdist']
                        tpdist = candle['TPdist']

                        trades_qty = self.openQty(candle,open_trades, filter_ticker, filter_strat)
                        if filter_ticker:
                            if filter_strat and strat['strategy'] in trades_qty[candle['Ticker']]:
                                trades_qty = trades_qty[candle['Ticker']][strat['strategy']]
                            else:
                                trades_qty = trades_qty[candle['Ticker']]
                        elif filter_strat:
                            trades_qty = trades_qty[strat['strategy']]

                        if strat['side'] != 0 and trades_qty < max_trades:

                            comission = self.config['commission'] * 100 if 'JPY' in candle['Ticker'] else self.config['commission']
                            size = risk/sldist*balance[-1]
                            if self.config['lots']:
                                size = size //10*10 if 'JPY' in candle['Ticker'] else size //1000*1000
                            
                            temp_max = self.pairs_config[strat['strategy']][candle['Ticker']]['max_size']
                            temp_min = self.pairs_config[strat['strategy']][candle['Ticker']]['min_size']
                            if size > temp_max:
                                size = temp_max
                            elif size < temp_min:
                                size = temp_min

                            if strat['side'] > 0:
                                # Buy order entry price
                                trade_order = self.pairs_config[strat['strategy']][candle['Ticker']]['order_type']
                                if self.pairs_config[strat['strategy']][candle['Ticker']]['order_type'] == 'market':
                                    if isinstance(strat['side'], float):
                                        entry = strat['side']
                                        if strat['side'] > candle['Close']:
                                            trade_order = 'stop'
                                        elif strat['side'] < candle['Close']:
                                            trade_order = 'limit'
                                    else:
                                        entry = prev_candle[candle['Ticker']]['Close'] + candle['Spread'] if candle['Open'] == prev_candle[candle['Ticker']]['Open'] else candle['Open'] + candle['Spread']
                                elif self.pairs_config[strat['strategy']][candle['Ticker']]['order_type'] == 'stop':
                                    entry = strat['side'] if isinstance(strat['side'],float) else prev_candle[candle['Ticker']]['High']
                                elif self.pairs_config[strat['strategy']][candle['Ticker']]['order_type'] == 'limit':
                                    entry = strat['side'] if isinstance(strat['side'],float) else prev_candle[candle['Ticker']]['Low']

                                # Check if the trade is already open
                                entered = False
                                for t in open_trades:
                                    if t['Entry'] == entry or\
                                        candle['Open'] == prev_candle[candle['Ticker']]['Open']:
                                        entered = True
                                        break

                                # Reset long open orders
                                if reset_orders:
                                    open_orders = [order for order in open_orders if (order['Signal'] != 'Buy') or (order['Ticker'] != candle['Ticker']) or (strat['strategy'] != order['Strategy'])]

                                # Define the new buy order
                                if not entered:
                                    trade = self.createTrade(candle, 'Buy', strat['strategy'], trade_order, entry, entry - sldist, 
                                                             entry + tpdist, comission, size, risk, balance[-1])

                                    # If market order execute it if not append to orders
                                    if trade_order == 'market':
                                        open_trades.append(trade)
                                    else:
                                        open_orders.append(trade)

                            if strat['side'] < 0:
                                # Sell order entry price
                                trade_order = self.pairs_config[strat['strategy']][candle['Ticker']]['order_type']
                                if self.pairs_config[strat['strategy']][candle['Ticker']]['order_type'] == 'market':
                                    if isinstance(strat['side'], float):
                                        entry = strat['side']
                                        if -strat['side'] > candle['Close']:
                                            trade_order = 'limit'
                                        elif -strat['side'] < candle['Close']:
                                            trade_order = 'stop'
                                    else:
                                        entry = prev_candle[candle['Ticker']]['Close'] if candle['Open'] == prev_candle[candle['Ticker']]['Open'] else candle['Open']
                                elif self.pairs_config[strat['strategy']][candle['Ticker']]['order_type'] == 'stop':
                                    entry = -strat['side'] if isinstance(strat['side'],float) else prev_candle[candle['Ticker']]['Low']
                                elif self.pairs_config[strat['strategy']][candle['Ticker']]['order_type'] == 'limit':
                                    entry = -strat['side'] if isinstance(strat['side'],float) else prev_candle[candle['Ticker']]['High']

                                # Check if the trade is already open
                                entered = False
                                for t in open_trades:
                                    if t['Entry'] == entry or\
                                        candle['Open'] == prev_candle[candle['Ticker']]['Open']:
                                        entered = True
                                        break
                                    
                                # Reset long open orders
                                if reset_orders:
                                    open_orders = [order for order in open_orders if (order['Signal'] != 'Sell') or (order['Ticker'] != candle['Ticker']) or (strat['strategy'] != order['Strategy'])]

                                # Define the new sell order
                                if not entered:
                                    trade = self.createTrade(candle, 'Sell', strat['strategy'], trade_order, entry, entry + sldist, 
                                                             entry - tpdist, comission, size, risk, balance[-1])

                                    # If market order execute it if not append to orders
                                    if trade_order == 'market':
                                        open_trades.append(trade)
                                    else:
                                        open_orders.append(trade)

                # Review pending orders execution
                if len(open_orders) > 0:

                    delete = []
                    for order in open_orders:

                        if order['Ticker'] == candle['Ticker']:

                            # STOP orders
                            if order['Order'] == 'stop':
                          
                                if order['Signal'] == 'Buy':

                                    if order['Entry'] <= candle['High'] + order['Spread']:
                                        order['EntryTime'] = candle['DateTime']
                                        order['Balance'] = balance[-1]
                                        open_trades.append(order)
                                        delete.append(order)

                                    elif candle['Low'] < order['SL']:
                                        #print(f"Buy Stop Cancelled by SL: \n Open - {candle['Open']} Low - {candle['Low']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)
                                    elif candle['High'] > order['TP']:
                                        #print(f"Buy Stop Cancelled by TP: \n Open - {candle['Open']} High - {candle['High']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)

                                if order['Signal'] == 'Sell':

                                    if order['Entry'] >= candle['Low']:
                                        order['EntryTime'] = candle['DateTime']
                                        order['Balance'] = balance[-1]
                                        open_trades.append(order)
                                        delete.append(order)

                                    elif candle['High'] > order['SL']:
                                        #print(f"Sell Stop Cancelled by SL: \n Open - {candle['Open']} High - {candle['High']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)
                                    elif candle['Low'] < order['TP']:
                                        #print(f"Sell Stop Cancelled by TP: \n Open - {candle['Open']} Low - {candle['Low']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)

                            # LIMIT orders
                            elif order['Order'] == 'limit':

                                if order['Signal'] == 'Sell':

                                    if order['Entry'] < candle['High']:
                                        order['EntryTime'] = candle['DateTime']
                                        order['Balance'] = balance[-1]
                                        open_trades.append(order)
                                        delete.append(order)
                                        
                                    elif candle['High'] > order['SL']:
                                        #print(f"Sell Limit Cancelled by SL: \n Open - {candle['Open']} High - {candle['High']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)
                                    elif candle['Low'] < order['TP']:
                                        #print(f"Sell Limit Cancelled by TP: \n Open - {candle['Open']} Low - {candle['Low']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)

                                if order['Signal'] == 'Buy':

                                    if order['Entry'] > candle['Low'] + order['Spread']:
                                        order['EntryTime'] = candle['DateTime']
                                        order['Balance'] = balance[-1]
                                        open_trades.append(order)
                                        delete.append(order)

                                    elif candle['Low'] < order['SL']:
                                        #print(f"Buy Limit Cancelled by SL: \n Open - {candle['Open']} Low - {candle['Low']} \
                                        #        \n SL - {order['SL']} Entry - {order['Entry']}  TP - {order['TP']}")
                                        delete.append(order)
                                    elif candle['High'] > order['TP']:
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

    config = {
        'capital': 10000.0,
        'init_date': '2010-01-01',
        'final_date': '2023-12-31',
        'lots': False,
        'commission': 0.000035,
    }


    strategies = {
        'TE':{
            'name': 'trendExplosion',
            'instruments': {
                #'BTCUSD': {'commission':100.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':0.0001, 'max_size':10000000},
                #'ETHUSD': {'commission':1.0, 'risk': 0.02, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':0.0001, 'max_size':10000000},
                'GBP_JPY': {'commission':0.01, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'NAS100_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'US30_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':4.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000}, #
                'SPX500_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':4.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000}, #
                'USD_JPY': {'commission':0.01, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                #'USTEC_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'XAU_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':4.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000000}, #
            },
            'use_sl': True,
            'use_tp': True,
            'time_limit': 50,
            'timeframe': 'H1',
        },
        'TC':{
            'name': 'trendContinuation',
            'instruments': {
                #'BTC_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 6.0, 'order_type':'stop', 'min_size':0.0001, 'max_size':10000000},
                #'ETH_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 6.0, 'order_type':'stop', 'min_size':0.0001, 'max_size':10000000},
                'GBP_JPY': {'commission':0.01, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 6.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'NAS100_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 6.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'SPX500_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 6.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'XAG_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 6.0, 'order_type':'stop', 'min_size':1, 'max_size':10000000},
                #'XBR_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 6.0, 'order_type':'stop', 'min_size':1000, 'max_size':10000000},
            },
            'use_sl': True,
            'use_tp': True,
            'time_limit': 200,
            'timeframe': 'H1',
        },
        'TKAMA':{
            'name': 'kamaTrend',
            'instruments': {
                'NAS100_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':1.0, 'TP': 4.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'US30_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':4.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
            },
            'use_sl': True,
            'use_tp': True,
            'time_limit': 100,
            'timeframe': 'H1',
        },
        'ATRE':{
            'name': 'atrExt',
            'instruments': {
                'NAS100_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'SPX500_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':4.0, 'TP': 8.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
            },
            'use_sl': True,
            'use_tp': True,
            'time_limit': 10,
            'timeframe': 'H1',
        },
        'TBO':{
            'name': 'turtlesBreakout',
            'instruments': {
                'GBP_JPY': {'commission':0.01, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'NAS100_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'US30_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'SPX500_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000},
                'XAG_USD': {'commission':0.0, 'risk': 0.01, 'volatility':True, 'SL':2.0, 'TP': 2.0, 'order_type':'stop', 'min_size':1, 'max_size':10000000},
            },
            'use_sl': True,
            'use_tp': True,
            'time_limit': 50,
            'timeframe': 'H1',
        },
    }



    # Prepare data needed for backtest
    oanda = Oanda(mode=False, token=open(os.path.join('KEYS','OANDA_DEMO_API.txt'),'r+').read())
    signals = Signals(backtest=True)
    indicators = Indicators()
    data = {}
    complete = []
    for strat in strategies:

        data[strat] = {}
        if strategies[strat]['name'] not in dir(signals):
            continue
        signal = getattr(signals, strategies[strat]['name'])

        for t in strategies[strat]['instruments']:
            temp = oanda.getCandles(instrument=t, timeframe=strategies[strat]['timeframe'],
                                    count=5000, df=True)
            temp.loc[:,'SLdist'] = strategies[strat]['instruments'][t]['SL'] * indicators.atr(n=20, method='s', dataname='ATR', new_df=temp)['ATR'] \
                                    if strategies[strat]['instruments'][t]['volatility'] else strategies[strat]['instruments'][t]['SL']
            temp.loc[:,'TPdist'] = strategies[strat]['instruments'][t]['TP'] * indicators.atr(n=20, method='s', dataname='ATR', new_df=temp)['ATR'] \
                                    if strategies[strat]['instruments'][t]['volatility'] else strategies[strat]['instruments'][t]['TP']
            temp.loc[:,'Ticker'] = [t]*len(temp)
            temp.loc[:,'Date'] = pd.to_datetime(temp.index)
            
            if 'Volume' not in temp.columns:
                temp['Volume'] = [0]*len(temp)

            temp = signal(df=temp, strat_name=strat)
            temp['Entry'] = temp[strat].apply(lambda x: [{'strategy':strat, 'side':x}])
            data[strat][t] = temp.copy()


        # Backtest
        bt = BackTest(pairs_config={strat: strategies[strat]['instruments']}, 
                    config={**config, **{k: strategies[strat][k] for k in strategies[strat] \
                                        if k not in ['name','instruments','timeframe']}})

        # Show main metrics
        complete_data = bt.dataPrepare(data[strat], unit='s')

        trades = bt.backtest(reset_orders=True)

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
