
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

    def __init__(self, data:pd.DataFrame, long:bool=True, short:bool=True, backtest:bool=True):

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
        self.long = long
        self.short = short
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
                                  (self.df['Close'].shift(1) > self.df['BBDN'].shift(1)) & \
                                    self.long, 
                                    1, 0)
        self.df['Short'] = np.where((self.df['Close'] > self.df['BBUP']) & \
                                    (self.df['Close'].shift(1) < self.df['BBUP'].shift(1)) & \
                                    self.short, 
                                    -1, 0)
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=[{'type':'trend','name':'BBUP'}, 
                                                              {'type':'trend','name':'BBDN'}])

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
                                   (self.df['Close'] < self.df['BBDN']+self.df['BBW']/2) & \
                                    self.long, 
                                    1, 0)
        self.df['Short'] = np.where((self.df['Close'] < self.df['BBUP']) & \
                                  (self.df['Close'].shift(1) > self.df['BBUP'].shift(1)) & \
                                   (self.df['Close'] > self.df['BBDN']+self.df['BBW']/2) & \
                                    self.short, 
                                    -1, 0)
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=[{'type':'trend','name':'BBUP'}, 
                                                              {'type':'trend','name':'BBDN'}])

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
                                        buy.append(1 if self.long else 0)
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
                                        short.append(-1 if self.short else 0)
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
                signalChart(self.df, asset=asset, indicators=[{'type':'oscillator','name':'PctBB'}])

        return self.df

    def rsiAggresive(self, n:int=14, lower:float=30.0, upper:float=70.0, perf:bool=True, 
                     plot:bool=True) -> pd.DataFrame:

        '''
        Buy when the RSI crosses downwards the lower level.

        Parameters
        ----------
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        perf: bool
            True to print backtest performance.
        plot: bool
            True to plot entries and exits from backtest.
        '''

        self.df = self.ind.rsi(n=n, method='s', datatype='Close', dataname='RSI')

        self.df['Buy'] = np.where((self.df['RSI'] < lower) & \
                                  (self.df['RSI'].shift(1) > lower) & \
                                    self.long, 
                                    1, 0)
        self.df['Short'] = np.where((self.df['RSI'] > upper) & \
                                  (self.df['RSI'].shift(1) < upper) & \
                                    self.short, 
                                    -1, 0)
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=[{'type':'oscillator','name':'RSI'}])

        return self.df
    
    def rsiConservative(self, n:int=14, lower:float=30.0, upper:float=70.0, perf:bool=True, 
                     plot:bool=True) -> pd.DataFrame:

        '''
        Buy when the RSI crosses upwards the lower level.

        Parameters
        ----------
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        perf: bool
            True to print backtest performance.
        plot: bool
            True to plot entries and exits from backtest.
        '''

        self.df = self.ind.rsi(n=n, method='s', datatype='Close', dataname='RSI')

        self.df['Buy'] = np.where((self.df['RSI'] > lower) & \
                                  (self.df['RSI'].shift(1) < lower) & \
                                    self.long, 
                                    1, 0)
        self.df['Short'] = np.where((self.df['RSI'] < upper) & \
                                  (self.df['RSI'].shift(1) > upper) & \
                                    self.short, 
                                    -1, 0)
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=[{'type':'oscillator','name':'RSI'}])

        return self.df
    
    def rsiCross(self, n:int=14, m:int=9, lower:float=40.0, upper:float=60.0, perf:bool=True, 
                     plot:bool=True) -> pd.DataFrame:

        '''
        Buy when the price crosses the Moving Average while the RSI is under the lower level.

        Parameters
        ----------
        n: int
            RSI period.
        m: int
            Moving Average period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        perf: bool
            True to print backtest performance.
        plot: bool
            True to plot entries and exits from backtest.
        '''

        self.df = self.ind.movingAverage(n=m, method='s', datatype='Close', dataname='MA')
        self.df = self.ind.rsi(n=n, method='s', datatype='Close', dataname='RSI')

        self.df['Buy'] = np.where((self.df['RSI'] < lower) & \
                                  (self.df['Close'].shift(1) < self.df['MA'].shift(1)) & \
                                  (self.df['Close'] > self.df['MA']) & \
                                  self.long, 
                                    1, 0)
        self.df['Short'] = np.where((self.df['RSI'] > upper) & \
                                  (self.df['Close'].shift(1) > self.df['MA'].shift(1)) & \
                                  (self.df['Close'] < self.df['MA']) & \
                                  self.short, 
                                    -1, 0)
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=[{'type':'oscillator','name':'RSI'},
                                                              {'type':'trend','name':'MA'}])

        return self.df

    def rsiDivergence(self, n:int=14, lower:float=40.0, upper:float=60.0, 
                    width:int=60, perf:bool=True, plot:bool=True) -> pd.DataFrame:

        '''
        Buy when the RSI makes a higher low below the lower level having 
        crossed it upwards after the previous low.

        Parameters
        ----------
        n: int
            RSI period.
        lower: float
            RSI lower limit.
        upper: float
            RSI upper limit.
        width: int
            Divergence width.
        perf: bool
            True to print backtest performance.
        plot: bool
            True to plot entries and exits from backtest.
        '''

        self.df = self.ind.rsi(n=n, method='s', datatype='Close', dataname='RSI')
        max_i = len(self.df)
        buy = []
        short = []
        for i,idx  in enumerate(self.df.index):
            candle = self.df.iloc[i]
            done = False

            # Long
            if candle['RSI'] < lower:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    high = self.df.iloc[j]
                    if lower < high['RSI'] and high['RSI'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            higher_low = self.df.iloc[k]
                            if higher_low['RSI'] < lower and \
                                higher_low['Close'] < candle['Close'] and \
                                higher_low['RSI'] > candle['RSI']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if lower < self.df.iloc[l]['RSI']:
                                        buy.append(1 if self.long else 0)
                                        short.append(0)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break

            # Short
            elif candle['RSI'] > upper:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    low = self.df.iloc[j]
                    if lower < low['RSI'] and low['RSI'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            lower_high = self.df.iloc[k]
                            if lower_high['RSI'] > upper and \
                                lower_high['Close'] > candle['Close'] and \
                                lower_high['RSI'] < candle['RSI']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if upper > self.df.iloc[l]['RSI']:
                                        buy.append(0)
                                        short.append(-1 if self.short else 0)
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
                signalChart(self.df, asset=asset, indicators=[{'type':'oscillator','name':'RSI'}])

        return self.df

    def rsiExtremeDuration(self, n:int=14, lower:float=30.0, upper:float=70.0, 
                           perf:bool=True, plot:bool=True) -> pd.DataFrame:

        '''
        Buy when the RSI crosses upwards the lower level after beeing for 5 periods below it.

        Parameters
        ----------
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        perf: bool
            True to print backtest performance.
        plot: bool
            True to plot entries and exits from backtest.
        '''

        self.df = self.ind.rsi(n=n, method='s', datatype='Close', dataname='RSI')

        self.df['Buy'] = np.where((self.df['RSI'] > lower) & \
                                  (self.df['RSI'].shift(1) < lower) & \
                                  (self.df['RSI'].shift(2) < lower) & \
                                  (self.df['RSI'].shift(3) < lower) & \
                                  (self.df['RSI'].shift(4) < lower) & \
                                  (self.df['RSI'].shift(5) < lower) & \
                                  self.long, 
                                    1, 0)
        self.df['Short'] = np.where((self.df['RSI'] < upper) & \
                                  (self.df['RSI'].shift(1) > upper) & \
                                  (self.df['RSI'].shift(2) > upper) & \
                                  (self.df['RSI'].shift(3) > upper) & \
                                  (self.df['RSI'].shift(4) > upper) & \
                                  (self.df['RSI'].shift(5) > upper) & \
                                  self.short, 
                                    -1, 0)
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=[{'type':'oscillator','name':'RSI'}])

        return self.df

    def rsiExtreme(self, n:int=14, lower:float=30.0, upper:float=70.0, 
                           perf:bool=True, plot:bool=True) -> pd.DataFrame:

        '''
        Buy when the RSI crosses the lower level just after crossing the upper level.

        Parameters
        ----------
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        perf: bool
            True to print backtest performance.
        plot: bool
            True to plot entries and exits from backtest.
        '''

        self.df = self.ind.rsi(n=n, method='s', datatype='Close', dataname='RSI')

        self.df['Buy'] = np.where((self.df['RSI'] < lower) & \
                                  (self.df['RSI'].shift(1) > lower) & \
                                  self.long, 
                                    1, 0)
        self.df['Short'] = np.where((self.df['RSI'] > upper) & \
                                  (self.df['RSI'].shift(1) < upper) & \
                                  self.short, 
                                    -1, 0)
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=[{'type':'oscillator','name':'RSI'}])

        return self.df
    
    def rsiM(self, n:int=5, lower:float=30.0, upper:float=70.0, 
            perf:bool=True, plot:bool=True) -> pd.DataFrame:

        '''
        Buy when the RSI when creates an M pattern surrounding the lower level.

        Parameters
        ----------
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        perf: bool
            True to print backtest performance.
        plot: bool
            True to plot entries and exits from backtest.
        '''

        self.df = self.ind.rsi(n=n, method='s', datatype='Close', dataname='RSI')

        self.df['Buy'] = np.where((self.df['RSI'] > lower) & \
                                  (self.df['RSI'].shift(1) < lower) & \
                                  (self.df['RSI'].shift(2) > lower) & \
                                  (self.df['RSI'].shift(3) < lower) & \
                                  (self.df['RSI'].shift(4) > lower) & \
                                  (self.df['RSI'].shift(1) < self.df['RSI'].shift(3)) & \
                                  self.long, 
                                    1, 0)
        self.df['Short'] = np.where((self.df['RSI'] < upper) & \
                                  (self.df['RSI'].shift(1) > upper) & \
                                  (self.df['RSI'].shift(2) < upper) & \
                                  (self.df['RSI'].shift(3) > upper) & \
                                  (self.df['RSI'].shift(4) < upper) & \
                                  (self.df['RSI'].shift(1) > self.df['RSI'].shift(3)) & \
                                  self.short, 
                                    -1, 0)
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=[{'type':'oscillator','name':'RSI'}])

        return self.df
    
    def rsiReversal(self, n:int=5, lower:float=30.0, upper:float=70.0, 
                    tolerance:float=3, perf:bool=True, plot:bool=True
                    ) -> pd.DataFrame:

        '''
        Buy when the RSI crosses upwards the lower level.

        Parameters
        ----------
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        perf: bool
            True to print backtest performance.
        plot: bool
            True to plot entries and exits from backtest.
        '''

        self.df = self.ind.rsi(n=n, method='s', datatype='Close', dataname='RSI')

        self.df['Buy'] = np.where((self.df['RSI'] > lower) & \
                                  (self.df['RSI'].shift(1) < lower) & \
                                  (self.df['RSI'].shift(2) > lower) & \
                                  (self.df['RSI'].shift(3) < lower) & \
                                  (self.df['RSI'].shift(4) > lower) & \
                                  (self.df['RSI'].shift(1) < self.df['RSI'].shift(3)) & \
                                  self.long, 
                                    1, 0)
        self.df['Short'] = np.where((self.df['RSI'] < upper) & \
                                  (self.df['RSI'].shift(1) > upper) & \
                                  (self.df['RSI'].shift(2) < upper) & \
                                  (self.df['RSI'].shift(3) > upper) & \
                                  (self.df['RSI'].shift(4) < upper) & \
                                  (self.df['RSI'].shift(1) > self.df['RSI'].shift(3)) & \
                                  self.short, 
                                    -1, 0)
        
        if self.backtest:
            self.df['Buy'] = self.df['Buy'].shift(1).fillna(0)
            self.df['Short'] = self.df['Short'].shift(1).fillna(0)

            if perf:
                self.df = performance(self.df)
            if plot:
                signalChart(self.df, asset=asset, indicators=[{'type':'oscillator','name':'RSI'}])

        return self.df



if __name__ == '__main__':

    asset = 'SPY'

    data = getData(asset, tf='1d')['data']

    pi = PrimaryIndicators(data, long=True, short=False, backtest=True)
    data = pi.rsiM(lower=40 ,upper=60 ,plot=True)

