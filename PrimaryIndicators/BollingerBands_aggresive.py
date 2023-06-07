
import os
import sys
sys.path.append('..')

import numpy as np
import pandas as pd

from utils import getData, performance, signalChart
from indicators import Indicators


class PrimaryIndicators:

    ''' 
    Class with one indicator contrarian strategies.
    '''

    def __init__(self, data:pd.DataFrame, backtest:bool=True):

        '''
        Function initiate the class.

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame with OHCL data for an asset. The open columns must 
            be named 'Open', the close column 'Close', the high column
            'High' and the low column 'Low
        backtest: bool
            True to shift the signals forward to make sure not to have 
            future bias. It gives the option to print metrics and plot charts.
        '''
        
        if 'Open' not in data.columns or 'High' not in data.columns or \
            'Low' not in data.columns or 'Close' not in data.columns:
            raise ValueError('Open, High, Low and Close must be between the dataframe columns.')
        
        self.df = data
        self.backtest = backtest
        self.ind = Indicators(data)

    def bollingerAggresive(self, n:int=20, desvi:float=2.0, perf:bool=True, plot:bool=True
                           ) -> pd.DataFrame:

        '''
        Buy when the price crosses downwards the lower bollinger band.

        Parameters
        ----------
        n: int
            Bollinger Bands period.
        desvi: float
            Bollinger Bands deviation.
        perf: bool
            True to print backtest performance.
        plot: bool
            True to plot entries and exits from backtest.
        '''

        self.df = self.ind.bollingerBands(n=n, method='s', desvi=desvi, datatype='Close', dataname='BB')

        self.df['Buy'] = np.where((self.df['Close'] < self.df['BBDN']) & \
                                  (self.df['Close'].shift(1) > self.df['BBDN'].shift(1)), 
                                    1, 0)
        self.df['Short'] = np.where((self.df['Close'] > self.df['BBUP']) & \
                                    (self.df['Close'].shift(1) < self.df['BBUP'].shift(1)), 
                                    1, 0)
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=['BBUP', 'BBDN'])

        return self.df

    def bollingerConservative(self, n:int=20, desvi:float=2.0, perf:bool=True, plot:bool=True
                           ) -> pd.DataFrame:

        '''
        Buy when the price crosses upwards the lower bollinger band.

        Parameters
        ----------
        n: int
            Bollinger Bands period.
        desvi: float
            Bollinger Bands deviation.
        perf: bool
            True to print backtest performance.
        plot: bool
            True to plot entries and exits from backtest.
        '''

        self.df = self.ind.bollingerBands(n=n, method='s', desvi=desvi, datatype='Close', dataname='BB')

        self.df['Buy'] = np.where((self.df['Close'] > self.df['BBDN']) & \
                                  (self.df['Close'].shift(1) < self.df['BBDN'].shift(1)) & \
                                   (self.df['Close'] < self.df['BBDN']+self.df['BBW']/2), 
                                    1, 0)
        self.df['Short'] = np.where((self.df['Close'] < self.df['BBUP']) & \
                                  (self.df['Close'].shift(1) > self.df['BBUP'].shift(1)) & \
                                   (self.df['Close'] > self.df['BBDN']+self.df['BBW']/2), 
                                    1, 0)
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=['BBUP', 'BBDN'])

        return self.df
    
    def bollingerDivergence(self, n:int=20, desvi:float=2.0, lower:float=0, upper:float=1, 
                            width:int=60, perf:bool=True, plot:bool=True) -> pd.DataFrame:

        '''
        Buy when the price makes a higher low below the lower bollinger band having 
        crossed it upwards after the previous low.

        Parameters
        ----------
        n: int
            Bollinger Bands period.
        desvi: float
            Bollinger Bands deviation.
        lower: float
            Value below which to look for upwards divergence.
        upper: float
            Value above which to look for downwards divergence.
        width: int
            Divergence width.
        perf: bool
            True to print backtest performance.
        plot: bool
            True to plot entries and exits from backtest.
        '''

        #self.df = self.ind.bollingerBands(n=n, method='s', desvi=desvi, datatype='Close', dataname='BB')
        self.df = self.ind.pctBollinger(n=n, desvi=desvi, datatype='Close', dataname='PctBB')
        max_i = len(self.df)
        buy = []
        short = []
        for i,idx  in enumerate(self.df.index):
            candle = self.df.iloc[i]
            done = False

            # Long
            if candle['PctBB'] < lower:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    high = self.df.iloc[j]
                    if lower < high['PctBB'] and high['PctBB'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            higher_low = self.df.iloc[k]
                            if higher_low['PctBB'] < lower and \
                                higher_low['Close'] < candle['Close'] and \
                                higher_low['PctBB'] > candle['PctBB']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if lower < self.df.iloc[l]['PctBB']:
                                        buy.append(1)
                                        short.append(0)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break

            # Short
            elif candle['PctBB'] > upper:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    low = self.df.iloc[j]
                    if lower < low['PctBB'] and low['PctBB'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            lower_high = self.df.iloc[k]
                            if lower_high['PctBB'] > upper and \
                                lower_high['Close'] > candle['Close'] and \
                                lower_high['PctBB'] < candle['PctBB']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if upper > self.df.iloc[l]['PctBB']:
                                        buy.append(0)
                                        short.append(1)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break
            
            if not done:
                buy.append(0)
                short.append(0)

        self.df['Buy'] = buy
        self.df['Short'] = short
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=['BBUP', 'BBDN'])

        return self.df



if __name__ == '__main__':

    asset = 'SPY'

    data = getData(asset, tf='1d')['data']

    pi = PrimaryIndicators(data, backtest=True)
    data = pi.bollingerDivergence(n=20, desvi=2.0, plot=True)

