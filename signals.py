
import numpy as np
import pandas as pd

from indicators import Indicators
from config import strategies


class PrimaryIndicatorSignals:

    def __init__(self,df:pd.DataFrame=None, backtest:bool=False):

        self.df = df
        self.indicators = Indicators(df)
        self.shift = 1 if backtest else 0

    def _newDf(self, df:pd.DataFrame):

        try:
            self.df = self.df.copy() if not isinstance(df, pd.DataFrame) else df
        except:
            print(df)
            raise(ValueError('Error trying to store the new DataFrame.'))
    
        if 'SLdist' not in self.df.columns:
            raise ValueError('"SLdist" is not a column from the dataframe.')
        
    def _checkStrategyConfig(self, strat_name):

        if strat_name not in list(strategies.keys()):
            raise ValueError(f'Strategy "{strat_name}" not between the tradeable ' + \
                             'list: '+','.join(list(strategies.keys())))

    def bollingerAggresive(self,df:pd.DataFrame=None, n:int=20, dev:float=2.0,
                        strat_name:str='BBAgr', exit_signal:bool=False
                        ) -> pd.DataFrame: 
        
        '''
        Buy when the price crosses downwards the lower bollinger band.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Bollinger Bands period.
        dev: float
            Bollinger Bands deviation.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.bollingerBands(n=n, method='s', desvi=dev, 
                                            datatype='Close', dataname='BB')

        short_condition = (df['Close'] > df['BBUP']) & \
                        (df['Close'].shift(1) < df['BBUP'].shift(1))
        long_condition = (df['Close'] < df['BBDN']) & \
                        (df['Close'].shift(1) > df['BBDN'].shift(1))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df
    
    def bollingerConservative(self,df:pd.DataFrame=None, n:int=20, dev:float=2.0,
                        strat_name:str='BBCons', exit_signal:bool=False
                        ) -> pd.DataFrame: 
        
        '''
        Buy when the price crosses upwards the lower bollinger band.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Bollinger Bands period.
        dev: float
            Bollinger Bands deviation.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.bollingerBands(n=n, method='s', desvi=dev, 
                                            datatype='Close', dataname='BB')

        short_condition = (df['Close'] < df['BBUP']) & \
                        (df['Close'].shift(1) > df['BBUP'].shift(1)) & \
                        (df['Close'] > df['BBDN']+df['BBW']/2)
        long_condition = (df['Close'] > df['BBDN']) & \
                        (df['Close'].shift(1) < df['BBDN'].shift(1)) & \
                        (df['Close'] < df['BBDN']+df['BBW']/2)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def bollingerDivergence(self,df:pd.DataFrame=None, n:int=20, dev:float=2.0,
                        lower:float=0, upper:float=1, width:int=60,
                        strat_name:str='BBDiv', exit_signal:bool=False
                        ) -> pd.DataFrame: 
        
        '''
        Buy when the price makes a higher low below the lower bollinger band having 
        crossed it upwards after the previous low.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Bollinger Bands period.
        dev: float
            Bollinger Bands deviation.
        lower: float
            Value below which to look for upwards divergence.
        upper: float
            Value above which to look for downwards divergence.
        width: int
            Divergence width.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.pctBollinger(n=n, desvi=dev, datatype='Close', 
                                          dataname='PctBB')
        
        max_i = len(df)
        long_condition = []
        short_condition = []
        for i,idx  in enumerate(df.index):
            candle = df.iloc[i]
            done = False

            # Long
            if candle['PctBB'] < lower:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    high = df.iloc[j]
                    if lower < high['PctBB'] and high['PctBB'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            higher_low = df.iloc[k]
                            if higher_low['PctBB'] < lower and \
                                higher_low['Close'] < candle['Close'] and \
                                higher_low['PctBB'] > candle['PctBB']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if lower < df.iloc[l]['PctBB']:
                                        long_condition.append(True)
                                        short_condition.append(False)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break

            # Short
            elif candle['PctBB'] > upper:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    low = df.iloc[j]
                    if lower < low['PctBB'] and low['PctBB'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            lower_high = df.iloc[k]
                            if lower_high['PctBB'] > upper and \
                                lower_high['Close'] > candle['Close'] and \
                                lower_high['PctBB'] < candle['PctBB']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if upper > df.iloc[l]['PctBB']:
                                        long_condition.append(False)
                                        short_condition.append(True)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break
            
            if not done:
                long_condition.append(False)
                short_condition.append(False)
                
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def rsiAggresive(self,df:pd.DataFrame=None, n:int=14, lower:float=30.0,
                     upper:float=70.0, strat_name:str='RSIAgr', 
                     exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI crosses downwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Bollinger Bands period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.rsi(n=n, method='s', datatype='Close', 
                                 dataname='RSI')

        short_condition = (df['RSI'] > upper) & (df['RSI'].shift(1) < upper)
        long_condition = (df['RSI'] < lower) & (df['RSI'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def rsiConservative(self,df:pd.DataFrame=None, n:int=14, lower:float=30.0,
                     upper:float=70.0, strat_name:str='RSICons', 
                     exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI crosses upwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.rsi(n=n, method='s', datatype='Close', 
                                 dataname='RSI')

        short_condition = (df['RSI'] < upper) & (df['RSI'].shift(1) > upper)
        long_condition = (df['RSI'] > lower) & (df['RSI'].shift(1) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def rsiCross(self,df:pd.DataFrame=None, n:int=14, m:int=9, lower:float=40.0,
                upper:float=60.0, strat_name:str='RSICross', 
                exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the price crosses the Moving Average while the RSI is 
        under the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.movingAverage(n=m, method='s', 
                                datatype='Close', dataname='MA')
        df = self.indicators.rsi(n=n, method='s', datatype='Close', 
                                 dataname='RSI')

        short_condition = (df['RSI'] > upper) & \
                        (df['Close'].shift(1) > df['MA'].shift(1)) & \
                        (df['Close'] < df['MA'])
        long_condition = (df['RSI'] < lower) & \
                        (df['Close'].shift(1) < df['MA'].shift(1)) & \
                        (df['Close'] > df['MA'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def rsiDivergence(self,df:pd.DataFrame=None, n:int=14, m:int=9, 
                      lower:float=40.0, upper:float=60.0, width:int=60,
                      strat_name:str='RSIDiv', exit_signal:bool=False
                      ) -> pd.DataFrame: 
        
        '''
        Buy when the RSI makes a higher low below the lower level having 
        crossed it upwards after the previous low.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        width: int
            Divergence width.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.rsi(n=n, method='s', datatype='Close', 
                                 dataname='RSI')

        max_i = len(df)
        long_condition = []
        short_condition = []
        for i,idx  in enumerate(df.index):
            candle = df.iloc[i]
            done = False

            # Long
            if candle['RSI'] < lower:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    high = df.iloc[j]
                    if lower < high['RSI'] and high['RSI'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            higher_low = df.iloc[k]
                            if higher_low['RSI'] < lower and \
                                higher_low['Close'] < candle['Close'] and \
                                higher_low['RSI'] > candle['RSI']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if lower < df.iloc[l]['RSI']:
                                        long_condition.append(True)
                                        short_condition.append(False)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break

            # Short
            elif candle['RSI'] > upper:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    low = df.iloc[j]
                    if lower < low['RSI'] and low['RSI'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            lower_high = df.iloc[k]
                            if lower_high['RSI'] > upper and \
                                lower_high['Close'] > candle['Close'] and \
                                lower_high['RSI'] < candle['RSI']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if upper > df.iloc[l]['RSI']:
                                        long_condition.append(False)
                                        short_condition.append(True)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break
            
            if not done:
                long_condition.append(False)
                short_condition.append(False)

        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def rsiExtremeDuration(self,df:pd.DataFrame=None, n:int=14,
                           lower:float=40.0, upper:float=60.0, 
                           strat_name:str='RSIExtDur', exit_signal:bool=False
                           ) -> pd.DataFrame: 
        
        '''
        Buy when the RSI crosses upwards the lower level after beeing 
        for 5 periods below it.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.rsi(n=n, method='s', datatype='Close', 
                                 dataname='RSI')

        short_condition = (df['RSI'] < upper) & \
                        (df['RSI'].shift(1) > upper) & \
                        (df['RSI'].shift(2) > upper) & \
                        (df['RSI'].shift(3) > upper) & \
                        (df['RSI'].shift(4) > upper) & \
                        (df['RSI'].shift(5) > upper)
        long_condition = (df['RSI'] > lower) & \
                        (df['RSI'].shift(1) < lower) & \
                        (df['RSI'].shift(2) < lower) & \
                        (df['RSI'].shift(3) < lower) & \
                        (df['RSI'].shift(4) < lower) & \
                        (df['RSI'].shift(5) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df
    
    def rsiExtreme(self,df:pd.DataFrame=None, n:int=14, lower:float=30.0, 
                   upper:float=70.0, strat_name:str='RSIExt',
                   exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI crosses the lower level just after crossing 
        the upper level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.rsi(n=n, method='s', datatype='Close', 
                                 dataname='RSI')

        short_condition = (df['RSI'] > upper) & \
                        (df['RSI'].shift(1) < upper)
        long_condition = (df['RSI'] < lower) & \
                        (df['RSI'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df
    
    def rsiM(self,df:pd.DataFrame=None, n:int=14, lower:float=30.0, 
                   upper:float=70.0, strat_name:str='RSIM',
                   exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI when creates an M pattern surrounding the 
        lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.rsi(n=n, method='s', datatype='Close', 
                                 dataname='RSI')

        short_condition = (df['RSI'] < upper) & \
                        (df['RSI'].shift(1) > upper) & \
                        (df['RSI'].shift(2) < upper) & \
                        (df['RSI'].shift(3) > upper) & \
                        (df['RSI'].shift(4) < upper) & \
                        (df['RSI'].shift(1) > df['RSI'].shift(3))
        long_condition = (df['RSI'] > lower) & \
                        (df['RSI'].shift(1) < lower) & \
                        (df['RSI'].shift(2) > lower) & \
                        (df['RSI'].shift(3) < lower) & \
                        (df['RSI'].shift(4) > lower) & \
                        (df['RSI'].shift(1) < df['RSI'].shift(3))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def rsiReversal(self,df:pd.DataFrame=None, n:int=14, lower:float=30.0, 
                   upper:float=70.0, tolerance:float=3, strat_name:str='RSIRev',
                   exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI crosses the lower level for just one period.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        tolerance: float
            Limits tolerance.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.rsi(n=n, method='s', datatype='Close', 
                                 dataname='RSI')

        short_condition = (df['RSI'] <= upper) & \
                        (df['RSI'].shift(1) >= upper-tolerance) & \
                        (df['RSI'].shift(2) <= df['RSI'])
        long_condition = (df['RSI'] >= lower) & \
                        (df['RSI'].shift(1) <= lower+tolerance) & \
                        (df['RSI'].shift(2) >= df['RSI'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def rsiPullback(self,df:pd.DataFrame=None, n:int=14, 
                    lower:float=30.0, upper:float=70.0, tolerance:float=3,
                    strat_name:str='RSIPull', exit_signal:bool=False
                    ) -> pd.DataFrame: 
        
        '''
        Buy when the RSI crosses by second time the lower limit.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        tolerance: float
            Limits tolerance.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.rsi(n=n, method='s', datatype='Close', 
                                 dataname='RSI')

        max_i = len(df)
        long_condition = []
        short_condition = []
        for i,idx  in enumerate(df.index):
            prev_candle = df.iloc[i-1]
            candle = df.iloc[i]
            done = False

            # Long
            if prev_candle['RSI'] < lower and lower < candle['RSI']:
                for j in range(i+1, len(df)):
                    prev_last = df.iloc[j-1]
                    last = df.iloc[j]
                    if lower <= last['RSI'] and last['RSI'] < lower+tolerance and \
                        prev_last['RSI'] > last['RSI']:
                        long_condition.append(True)
                        short_condition.append(False)
                        done = True
                        break
                    if done:
                        break

            # Short
            elif prev_candle['RSI'] > upper and upper > candle['RSI']:
                for j in range(i+1, len(df)):
                    prev_last = df.iloc[j-1]
                    last = df.iloc[j]
                    if upper >= last['RSI'] and last['RSI'] > upper+tolerance and \
                        prev_last['RSI'] < last['RSI']:
                        long_condition.append(False)
                        short_condition.append(True)
                        done = True
                        break
                    if done:
                        break
            
            if not done:
                long_condition.append(False)
                short_condition.append(False)

        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def rsiV(self,df:pd.DataFrame=None, n:int=5, lower:float=30.0, 
            upper:float=70.0, strat_name:str='RSIRev',
            exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI when creates a V pattern surrounding the 
        lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.rsi(n=n, method='s', datatype='Close', 
                                 dataname='RSI')

        short_condition = (df['RSI'] < upper) & \
                        (df['RSI'].shift(1) > upper) & \
                        (df['RSI'].shift(2) < upper)
        long_condition = (df['RSI'] > lower) & \
                        (df['RSI'].shift(1) < lower) & \
                        (df['RSI'].shift(2) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def stochAggresive(self,df:pd.DataFrame=None, n:int=14, lower:float=15.0,
                     upper:float=85.0, strat_name:str='StochAgr', 
                     exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Stochastic crosses downwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Stochastic Oscillator period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
        
        df = self.indicators.stochasticOscillator(n=n, dataname='SO')

        short_condition = (df['SO'] > upper) & (df['SO'].shift(1) < upper)
        long_condition = (df['SO'] < lower) & (df['SO'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def stochConservative(self,df:pd.DataFrame=None, n:int=14, lower:float=15.0,
                     upper:float=85.0, strat_name:str='StochCons', 
                     exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Stochastic crosses upwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Stochastic Oscillator period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
        
        df = self.indicators.stochasticOscillator(n=n, dataname='SO')

        short_condition = (df['SO'] < upper) & (df['SO'].shift(1) > upper)
        long_condition = (df['SO'] > lower) & (df['SO'].shift(1) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def stochCross(self,df:pd.DataFrame=None, n:int=14, m:int=9, 
                   lower:float=15.0, upper:float=85.0, strat_name:str='StochCross', 
                   exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Stochastic crosses it's moving average while oversold.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Stochastic Oscillator period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
        
        df = self.indicators.stochasticOscillator(n=n, m=m, p=3, dataname='SO')

        short_condition = (df['SOD'] < df['SOK']) & \
                        (df['SOD'].shift(1) > df['SOK'].shift(1)) & \
                        (df['SOD'] > upper)
        long_condition = (df['SOD'] > df['SOK']) & \
                        (df['SOD'].shift(1) < df['SOK'].shift(1)) & \
                        (df['SOD'] < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def stochDivergence(self,df:pd.DataFrame=None, n:int=14, 
                      lower:float=15.0, upper:float=85.0, width:int=60,
                      strat_name:str='StochDiv', exit_signal:bool=False
                      ) -> pd.DataFrame: 
        
        '''
        Buy when the Stochastic Oscillator makes a higher low below the 
        lower level having crossed it upwards after the previous low.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        width: int
            Divergence width.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, dataname='SO')

        max_i = len(df)
        long_condition = []
        short_condition = []
        for i,idx  in enumerate(df.index):
            candle = df.iloc[i]
            done = False

            # Long
            if candle['SOD'] < lower:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    high = df.iloc[j]
                    if lower < high['SOD'] and high['SOD'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            higher_low = df.iloc[k]
                            if higher_low['SOD'] < lower and \
                                higher_low['Close'] < candle['Close'] and \
                                higher_low['SOD'] > candle['SOD']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if lower < df.iloc[l]['SOD']:
                                        long_condition.append(True)
                                        short_condition.append(False)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break

            # Short
            elif candle['SOD'] > upper:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    low = df.iloc[j]
                    if lower < low['SOD'] and low['SOD'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            lower_high = df.iloc[k]
                            if lower_high['SOD'] > upper and \
                                lower_high['Close'] > candle['Close'] and \
                                lower_high['SOD'] < candle['SOD']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if upper > df.iloc[l]['SOD']:
                                        long_condition.append(False)
                                        short_condition.append(True)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break
            
            if not done:
                long_condition.append(False)
                short_condition.append(False)

        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def stochExtremeDuration(self,df:pd.DataFrame=None, n:int=14,
                           lower:float=20.0, upper:float=80.0, 
                           strat_name:str='StochExtDur', exit_signal:bool=False
                           ) -> pd.DataFrame: 
        
        '''
        Buy when the Stochastic Oscillator crosses upwards the lower 
        level after beeing for 5 periods below it.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Stochastic Oscillator period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, dataname='SO')

        short_condition = (df['SOD'] < upper) & \
                        (df['SOD'].shift(1) > upper) & \
                        (df['SOD'].shift(2) > upper) & \
                        (df['SOD'].shift(3) > upper) & \
                        (df['SOD'].shift(4) > upper) & \
                        (df['SOD'].shift(5) > upper)
        long_condition = (df['SOD'] > lower) & \
                        (df['SOD'].shift(1) < lower) & \
                        (df['SOD'].shift(2) < lower) & \
                        (df['SOD'].shift(3) < lower) & \
                        (df['SOD'].shift(4) < lower) & \
                        (df['SOD'].shift(5) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df
    
    def stochExtreme(self,df:pd.DataFrame=None, n:int=2, lower:float=15.0, 
                   upper:float=85.0, strat_name:str='StochExt',
                   exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Stochastic Oscillator crosses the lower level just 
        after crossing the upper level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Stochastic Oscillator period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, dataname='SO')

        short_condition = (df['SOD'] > upper) & \
                        (df['SOD'].shift(1) < upper)
        long_condition = (df['SOD'] < lower) & \
                        (df['SOD'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df
    
    def stochM(self,df:pd.DataFrame=None, n:int=14, lower:float=20.0, 
                   upper:float=80.0, strat_name:str='StochM',
                   exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Stochastic Oscillator when creates an M pattern 
        surrounding the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Stochastic Oscillator period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, dataname='SO')

        short_condition = (df['SOD'] < upper) & \
                        (df['SOD'].shift(1) > upper) & \
                        (df['SOD'].shift(2) < upper) & \
                        (df['SOD'].shift(3) > upper) & \
                        (df['SOD'].shift(4) < upper) & \
                        (df['SOD'].shift(1) <= df['SOD'].shift(3))
        long_condition = (df['SOD'] > lower) & \
                        (df['SOD'].shift(1) < lower) & \
                        (df['SOD'].shift(2) > lower) & \
                        (df['SOD'].shift(3) < lower) & \
                        (df['SOD'].shift(4) > lower) & \
                        (df['SOD'].shift(1) >= df['SOD'].shift(3))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df

    def stochPullback(self,df:pd.DataFrame=None, n:int=14, 
                    lower:float=30.0, upper:float=70.0, tolerance:float=3,
                    strat_name:str='StochPull', exit_signal:bool=False
                    ) -> pd.DataFrame: 
        
        '''
        Buy when the RSI crosses by second time the lower limit.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        tolerance: float
            Limits tolerance.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, dataname='SO')

        max_i = len(df)
        long_condition = []
        short_condition = []
        for i,idx  in enumerate(df.index):
            prev_candle = df.iloc[i-1]
            candle = df.iloc[i]
            done = False

            # Long
            if prev_candle['SOD'] < lower and lower < candle['SOD']:
                for j in range(i+1, len(df)):
                    prev_last = df.iloc[j-1]
                    last = df.iloc[j]
                    if lower <= last['SOD'] and last['SOD'] < lower+tolerance and \
                        prev_last['SOD'] > last['SOD']:
                        long_condition.append(True)
                        short_condition.append(False)
                        done = True
                        break
                    if done:
                        break

            # Short
            elif prev_candle['SOD'] > upper and upper > candle['SOD']:
                for j in range(i+1, len(df)):
                    prev_last = df.iloc[j-1]
                    last = df.iloc[j]
                    if upper >= last['SOD'] and last['SOD'] > upper+tolerance and \
                        prev_last['SOD'] < last['SOD']:
                        long_condition.append(False)
                        short_condition.append(True)
                        done = True
                        break
                    if done:
                        break
            
            if not done:
                long_condition.append(False)
                short_condition.append(False)

        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df






class Signals(PrimaryIndicatorSignals):

    def __init__(self,df:pd.DataFrame=None, backtest:bool=False):

        self.df = df
        self.indicators = Indicators(df)
        self.shift = 1 if backtest else 0

    def _newDf(self, df:pd.DataFrame):

        try:
            self.df = self.df.copy() if not isinstance(df, pd.DataFrame) else df
        except:
            print(df)
            raise(ValueError('Error trying to store the new DataFrame.'))
    
        if 'SLdist' not in self.df.columns:
            raise ValueError('"SLdist" is not a column from the dataframe.')
        
    def _checkStrategyConfig(self, strat_name):

        if strat_name not in list(strategies.keys()):
            raise ValueError(f'Strategy "{strat_name}" not between the tradeable ' + \
                             'list: '+','.join(list(strategies.keys())))
    
    def turtlesBreakout(self,df:pd.DataFrame=None, 
                      n:int=100, strat_name:str='TBO') -> pd.DataFrame: 
        
        '''
        Calculates buy and sell signals.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        ma: int
            Moving Average period.
        dc_n: int
            Donchian Channel period.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''

        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.donchian(n, high_data='High', low_data='Low', dataname='DC', new_df=df)
        
        long_condition = (df['Close'] > df['DCUP'].shift(1))
        short_condition = (df['Close'] < df['DCDN'].shift(1))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        self.df = df.copy()

        return self.df
    
    def trendExplosion(self,df:pd.DataFrame=None, ma:int=20, dc_n:int=50, kc_mult:float=3.0,
                      strat_name:str='TE', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Calculates buy and sell signals.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        ma: int
            Moving Average period.
        dc_n: int
            Donchian Channel period.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()

        df.loc[:,'DCUP'] = df['High'].rolling(dc_n).max()
        df.loc[:,'DCDN'] = df['Low'].rolling(dc_n).min()

        df = self.indicators.atr(new_df=df, n=ma, dataname='ATR', tr=False)
        df.loc[:,'EMA1'] = df['Close'].ewm(span=ma, adjust=False).mean()
        df.loc[:,'KCH'] = df['EMA1'] + kc_mult*df['ATR']
        df.loc[:,'KCL'] = df['EMA1'] - kc_mult*df['ATR']

        long_condition = (df['High'] == df['DCUP']) & \
                        (df['Close'] >= df['KCH'])
        short_condition = (df['Low'] == df['DCDN']) & \
                        (df['Close'] <= df['KCL'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) < df['KCH2']) & \
                                  (df['Close'].shift(2) > df['KCH2'].shift(2)), 1,
                        np.where((df['Close'].shift(1) > df['KCL2']) & \
                                 (df['Close'].shift(2) < df['KCL2'].shift(2)), -1, 0))

        self.df = df

        return self.df
    
    def trendContinuation(self,df:pd.DataFrame=None, ma_1:int=20, ma_2:int=50, 
                          ma_3:int=100, strat_name:str='TC', 
                          exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Calculates buy and sell signals.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        ma_1: int
            Short exponential moving average period.
        ma_2: int
            Medium exponential moving average period.
        ma_3: int
            Long exponential moving average period.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df.loc[:,'EMA1'] = df['Close'].ewm(span=ma_1, adjust=False).mean()#.rolling(5).mean()
        df.loc[:,'EMA2'] = df['Close'].ewm(span=ma_2, adjust=False).mean()#.rolling(5).mean()
        df.loc[:,'EMA3'] = df['Close'].ewm(span=ma_3, adjust=False).mean()#.rolling(5).mean()

        df.loc[:,'STD'] = df['Close'].rolling(20).std()
        df.loc[:,'STDMA'] = df['STD'].rolling(10).mean()

        df = self.indicators.atr(n=20, dataname='ATR', tr=False, new_df=df)
        df.loc[:,'KCH1'] = df['EMA1'] + df['ATR']
        df.loc[:,'KCL1'] = df['EMA1'] - df['ATR']
        df.loc[:,'KCH2'] = df['EMA1'] + 3*df['ATR']
        df.loc[:,'KCL2'] = df['EMA1'] - 3*df['ATR']

        long_condition = (df['Close'] >= df['KCH1']) & \
                        (df['EMA2'] > df['EMA3'])
        short_condition = (df['Close'] <= df['KCL1']) & \
                        (df['EMA2'] < df['EMA3'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) < df['KCH2']) & (df['Close'].shift(2) > df['KCH2'].shift(2)), 1,
                        np.where((df['Close'].shift(1) > df['KCL2']) & (df['Close'].shift(2) < df['KCL2'].shift(2)), -1, 0))

        self.df = df

        return self.df
        
    def kamaTrend(self,df:pd.DataFrame=None, n_1:int=2, n_2:int=10, strat_name:str='TKAMA', 
                  exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Calculates buy and sell signals.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n_1: int
            Short KAMA period.
        n_2: int
            Long KAMA period.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
        
        df = self.indicators.kama(n=n_1, scf=2, scs=30, datatype='Close', dataname='KAMA', new_df=df)
        df = self.indicators.kama(n=n_2, scf=2, scs=30, datatype='Close', dataname='KAMA2', new_df=df)
        df['KAMAslope'] = (df['KAMA']-df['KAMA'].shift(2))/2
        df['KAMA2slope'] = (df['KAMA2']-df['KAMA2'].shift(2))/2

        long_condition = (df['KAMA'].shift(1) < df['KAMA2'].shift(1)) & \
                        (df['KAMA'] >= df['KAMA2']) & \
                        (df['KAMAslope'] > df['KAMA2slope'])
        short_condition = (df['KAMA'].shift(1) > df['KAMA2'].shift(1)) & \
                        (df['KAMA'] <= df['KAMA2']) & \
                        (df['KAMAslope'] < df['KAMA2slope'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df
        
    def atrExt(self,df:pd.DataFrame=None, n:int=20, quantile:float=0.9,
               strat_name:str='ATRE', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Calculates buy and sell signals.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            ATR period.
        quantile: float
            Quantile percentage.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._checkStrategyConfig(strat_name)
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.atr(n=n, method='s', dataname='ATR', new_df=df)
        df['ATRpct'] = df['ATR'].rolling(n).quantile(quantile)

        short_condition = (df['Close'] < df['Open']) & \
                        (df['ATR'] >= df['ATRpct'])
        long_condition = (df['Close'] > df['Open']) & \
                        (df['ATR'] >= df['ATRpct'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)

        df[strat_name] = np.where(exe_condition.shift(self.shift) & \
                                  long_condition.shift(self.shift), 1,
                        np.where(exe_condition.shift(self.shift) & \
                                short_condition.shift(self.shift), -1, 
                        0))

        if exit_signal:
            df['Exit'] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.df = df

        return self.df
    


if __name__ == '__main__':

    import os
    from oanda import Oanda

    oanda = Oanda(mode=False, 
                token=open(os.path.join('KEYS','OANDA_DEMO_API.txt'),'r+').read())

    raw = oanda.getCandles('BTC_USD','H1')
    raw = raw[raw['Complete']]
    raw['SLdist'] = Indicators(raw).atr(n=20, method='s', dataname='ATR')['ATR']

    signals = Signals(raw, backtest=False)
    data = signals.trendExplosion()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0, row_heights=[5,2])

    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)

    # Canlesticks
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], 
                        low=data['Low'], close=data['Close'], name='Price'))
    
    strat_name = [c for c in data.columns if 'Signal' in c]
    for s in strat_name:
        fig.add_trace(go.Scatter(x=data.index[data[s] > 0], y=data['Open'].shift(-1)[data[s] > 0], name='Long', 
                                 marker_color='Blue', marker_symbol='triangle-right', marker_size=15, mode='markers'), 
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index[data[s] < 0], y=data['Open'].shift(-1)[data[s] < 0], name='Short', 
                                 marker_color='Orange', marker_symbol='triangle-right', marker_size=15, mode='markers'), 
                      row=1, col=1)

    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        title_text=f'Price',
        autosize=False,
        width=900,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    fig.show()
