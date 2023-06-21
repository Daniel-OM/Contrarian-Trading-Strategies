
import numpy as np
import pandas as pd

from indicators import Indicators

    
# def checkStrategyConfig(self, strat_name):

#     from config import strategies

#     if strat_name not in list(strategies.keys()):
#         raise ValueError(f'Strategy "{strat_name}" not between the tradeable ' + \
#                             'list: '+','.join(list(strategies.keys())))

class OscillatorSignals:

    def __init__(self,df:pd.DataFrame=None, backtest:bool=False):

        self._newDf(df)
        self.indicators = Indicators(df)
        self.shift = 1 if backtest else 0

    def _newDf(self, df:pd.DataFrame, errors:bool=True) -> None:

        try:
            self.df = self.df.copy() if not isinstance(df, pd.DataFrame) else df
            if 'Spread' not in df:
                if errors:
                    raise ValueError('"Spread" is not between the dataframe columns.')
                else:
                    self.df['Spread'] = [0]*len(self.df)
                    print('"Spread" is not between the dataframe columns.')
            if 'SLdist' not in df:
                if errors:
                    raise ValueError('"SLdist" is not between the dataframe columns.')
                else:
                    self.df['SLdist'] = [0]*len(self.df)
                    print('"SLdist" is not between the dataframe columns.')

        except:
            print(df)
            raise(ValueError('Error trying to store the new DataFrame.'))
        
    def getIndicators(self):

        return [i for i in dir(self.indicators) if '_' not in i]

    def _checkIndicator(self, ind:str):

        if ind not in self.getIndicators():
            raise ValueError('Enter a valid indicator. You can check them calling \
                             the getIndicators() function.') 
    
    def _kwargsError(self, kwargs:dict, ind_name:str):

        if len(kwargs) <= 0:
            raise ValueError(f'No arguments for the indicator where given, check \
                            OscillatorSignals.indicators.{ind_name}() to get the \
                            needed arguments. At least the dataname should be given')
    
    def aggresive(self,df:pd.DataFrame=None, lower:float=30.0,
                upper:float=70.0, ind_name:str='rsi', exit_signal:bool=False, 
                **kwargs) -> pd.DataFrame: 
        
        '''
        Buy when the indicator crosses downwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        ind_name: str
            Name of the indicator that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.
        kwargs
            key - value pairs corresponding to the indicator arguments.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._kwargsError(kwargs, ind_name)
        strat_name = ind_name + 'Agr'
        self._newDf(df)
        df = self.df.copy()

        df = getattr(self.indicators, ind_name)(**kwargs)
        
        ind = kwargs['dataname']
        short_condition = (df[ind] > upper) & (df[ind].shift(1) < upper)
        long_condition = (df[ind] < lower) & (df[ind].shift(1) > lower)
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

    def conservative(self,df:pd.DataFrame=None, lower:float=30.0,
                     upper:float=70.0, ind_name:str='rsi', 
                     exit_signal:bool=False, **kwargs) -> pd.DataFrame: 
        
        '''
        Buy when the indicator crosses upwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        ind_name: str
            Name of the indicator that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._kwargsError(kwargs, ind_name)
        strat_name = ind_name + 'Cons'
        self._newDf(df)
        df = self.df.copy()

        df = getattr(self.indicators, ind_name)(**kwargs)

        ind = kwargs['dataname']
        short_condition = (df[ind] < upper) & (df[ind].shift(1) > upper)
        long_condition = (df[ind] > lower) & (df[ind].shift(1) < lower)
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

    def cross(self,df:pd.DataFrame=None, n:int=9, lower:float=40.0,
                upper:float=60.0, ind_name:str='rsi', 
                exit_signal:bool=False, **kwargs) -> pd.DataFrame: 
        
        '''
        Buy when the price crosses the Moving Average while the indicator is 
        under the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Moving Average period.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        ind_name: str
            Name of the indicator that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._kwargsError(kwargs, ind_name)
        strat_name = ind_name + 'Cross'
        self._newDf(df)
        df = self.df.copy()

        df = getattr(self.indicators, ind_name)(**kwargs)
        df = self.indicators.movingAverage(n=n, method='s', 
                                datatype='Close', dataname='MA')

        ind = kwargs['dataname']
        short_condition = (df[ind] > upper) & \
                        (df['Close'].shift(1) > df['MA'].shift(1)) & \
                        (df['Close'] < df['MA'])
        long_condition = (df[ind] < lower) & \
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

    def divergence(self,df:pd.DataFrame=None, 
                      lower:float=40.0, upper:float=60.0, width:int=60,
                      ind_name:str='rsi', exit_signal:bool=False,
                      **kwargs) -> pd.DataFrame: 
        
        '''
        Buy when the indicator makes a higher low below the lower level having 
        crossed it upwards after the previous low.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        width: int
            Divergence width.
        ind_name: str
            Name of the indicator that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._kwargsError(kwargs, ind_name)
        strat_name = ind_name + 'Div'
        self._newDf(df)
        df = self.df.copy()

        df = getattr(self.indicators, ind_name)(**kwargs)

        ind = kwargs['dataname']
        max_i = len(df)
        long_condition = []
        short_condition = []
        for i,idx  in enumerate(df.index):
            candle = df.iloc[i]
            done = False

            # Long
            if candle[ind] < lower:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    high = df.iloc[j]
                    if lower < high[ind] and high[ind] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            higher_low = df.iloc[k]
                            if higher_low[ind] < lower and \
                                higher_low['Close'] < candle['Close'] and \
                                higher_low[ind] > candle[ind]:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if lower < df.iloc[l][ind]:
                                        long_condition.append(True)
                                        short_condition.append(False)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break

            # Short
            elif candle[ind] > upper:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    low = df.iloc[j]
                    if lower < low[ind] and low[ind] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            lower_high = df.iloc[k]
                            if lower_high[ind] > upper and \
                                lower_high['Close'] > candle['Close'] and \
                                lower_high[ind] < candle[ind]:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if upper > df.iloc[l][ind]:
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

    def extremeDuration(self,df:pd.DataFrame=None,
                           lower:float=40.0, upper:float=60.0, 
                           ind_name:str='rsi', exit_signal:bool=False,
                           **kwargs) -> pd.DataFrame: 
        
        '''
        Buy when the indicator crosses upwards the lower level after beeing 
        for 5 periods below it.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        ind_name: str
            Name of the indicator that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._kwargsError(kwargs, ind_name)
        strat_name = ind_name + 'ExtDur'
        self._newDf(df)
        df = self.df.copy()

        df = getattr(self.indicators, ind_name)(**kwargs)

        ind = kwargs['dataname']

        short_condition = (df[ind] < upper) & \
                        (df[ind].shift(1) > upper) & \
                        (df[ind].shift(2) > upper) & \
                        (df[ind].shift(3) > upper) & \
                        (df[ind].shift(4) > upper) & \
                        (df[ind].shift(5) > upper)
        long_condition = (df[ind] > lower) & \
                        (df[ind].shift(1) < lower) & \
                        (df[ind].shift(2) < lower) & \
                        (df[ind].shift(3) < lower) & \
                        (df[ind].shift(4) < lower) & \
                        (df[ind].shift(5) < lower)
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
    
    def extreme(self,df:pd.DataFrame=None, lower:float=30.0, 
                   upper:float=70.0, ind_name:str='rsi',
                   exit_signal:bool=False, **kwargs) -> pd.DataFrame: 
        
        '''
        Buy when the indicator crosses the lower level just after crossing 
        the upper level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        ind_name: str
            Name of the indicator that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._kwargsError(kwargs, ind_name)
        strat_name = ind_name + 'Ext'
        self._newDf(df)
        df = self.df.copy()

        df = getattr(self.indicators, ind_name)(**kwargs)

        ind = kwargs['dataname']

        short_condition = (df[ind] > upper) & \
                        (df[ind].shift(1) < upper)
        long_condition = (df[ind] < lower) & \
                        (df[ind].shift(1) > lower)
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
    
    def mPattern(self,df:pd.DataFrame=None, lower:float=30.0, 
                   upper:float=70.0, ind_name:str='rsi',
                   exit_signal:bool=False, **kwargs) -> pd.DataFrame: 
        
        '''
        Buy when the indicator when creates an M pattern surrounding the 
        lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        ind_name: str
            Name of the indicator that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._kwargsError(kwargs, ind_name)
        strat_name = ind_name + 'M'
        self._newDf(df)
        df = self.df.copy()

        df = getattr(self.indicators, ind_name)(**kwargs)

        ind = kwargs['dataname']

        short_condition = (df[ind] < upper) & \
                        (df[ind].shift(1) > upper) & \
                        (df[ind].shift(2) < upper) & \
                        (df[ind].shift(3) > upper) & \
                        (df[ind].shift(4) < upper) & \
                        (df[ind].shift(1) > df[ind].shift(3))
        long_condition = (df[ind] > lower) & \
                        (df[ind].shift(1) < lower) & \
                        (df[ind].shift(2) > lower) & \
                        (df[ind].shift(3) < lower) & \
                        (df[ind].shift(4) > lower) & \
                        (df[ind].shift(1) < df[ind].shift(3))
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

    def reversal(self,df:pd.DataFrame=None, lower:float=30.0, 
                   upper:float=70.0, tolerance:float=3, ind_name:str='rsi',
                   exit_signal:bool=False, **kwargs) -> pd.DataFrame: 
        
        '''
        Buy when the indicator crosses the lower level for just one period.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        tolerance: float
            Limits tolerance.
        ind_name: str
            Name of the indicator that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._kwargsError(kwargs, ind_name)
        strat_name = ind_name + 'Rev'
        self._newDf(df)
        df = self.df.copy()

        df = getattr(self.indicators, ind_name)(**kwargs)

        ind = kwargs['dataname']

        short_condition = (df[ind] <= upper) & \
                        (df[ind].shift(1) >= upper-tolerance) & \
                        (df[ind].shift(2) <= df[ind])
        long_condition = (df[ind] >= lower) & \
                        (df[ind].shift(1) <= lower+tolerance) & \
                        (df[ind].shift(2) >= df[ind])
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

    def pullback(self,df:pd.DataFrame=None, 
                    lower:float=30.0, upper:float=70.0, tolerance:float=3,
                    ind_name:str='rsi', exit_signal:bool=False, **kwargs
                    ) -> pd.DataFrame: 
        
        '''
        Buy when the indicator crosses by second time the lower limit.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        tolerance: float
            Limits tolerance.
        ind_name: str
            Name of the indicator that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._kwargsError(kwargs, ind_name)
        strat_name = ind_name + 'Pull'
        self._newDf(df)
        df = self.df.copy()

        df = getattr(self.indicators, ind_name)(**kwargs)

        ind = kwargs['dataname']

        max_i = len(df)
        long_condition = []
        short_condition = []
        for i,idx  in enumerate(df.index):
            prev_candle = df.iloc[i-1]
            candle = df.iloc[i]
            done = False

            # Long
            if prev_candle[ind] < lower and lower < candle[ind]:
                for j in range(i+1, len(df)):
                    prev_last = df.iloc[j-1]
                    last = df.iloc[j]
                    if lower <= last[ind] and last[ind] < lower+tolerance and \
                        prev_last[ind] > last[ind]:
                        long_condition.append(True)
                        short_condition.append(False)
                        done = True
                        break
                    if done:
                        break

            # Short
            elif prev_candle[ind] > upper and upper > candle[ind]:
                for j in range(i+1, len(df)):
                    prev_last = df.iloc[j-1]
                    last = df.iloc[j]
                    if upper >= last[ind] and last[ind] > upper+tolerance and \
                        prev_last[ind] < last[ind]:
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

    def vPattern(self,df:pd.DataFrame=None, lower:float=30.0, 
            upper:float=70.0, ind_name:str='rsi',
            exit_signal:bool=False, **kwargs) -> pd.DataFrame: 
        
        '''
        Buy when the indicator when creates a V pattern surrounding the 
        lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        lower: float
            Lower limit.
        upper: float
            Upper limit.
        ind_name: str
            Name of the indicator that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        self._kwargsError(kwargs, ind_name)
        strat_name = ind_name + 'V'
        self._newDf(df)
        df = self.df.copy()

        df = getattr(self.indicators, ind_name)(**kwargs)

        ind = kwargs['dataname']

        short_condition = (df[ind] < upper) & \
                        (df[ind].shift(1) > upper) & \
                        (df[ind].shift(2) < upper)
        long_condition = (df[ind] > lower) & \
                        (df[ind].shift(1) < lower) & \
                        (df[ind].shift(2) > lower)
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




class PrimaryIndicatorSignals:

    def __init__(self,df:pd.DataFrame=None, backtest:bool=False):

        self.df = df
        self.indicators = Indicators(df)
        self.shift = 1 if backtest else 0

    def _newDf(self, df:pd.DataFrame, errors:bool=True) -> None:

        try:
            self.df = self.df.copy() if not isinstance(df, pd.DataFrame) else df
            if 'Spread' not in df:
                if errors:
                    raise ValueError('"Spread" is not between the dataframe columns.')
                else:
                    self.df['Spread'] = [0]*len(self.df)
                    print('"Spread" is not between the dataframe columns.')
            if 'SLdist' not in df:
                if errors:
                    raise ValueError('"SLdist" is not between the dataframe columns.')
                else:
                    self.df['SLdist'] = [0]*len(self.df)
                    print('"SLdist" is not between the dataframe columns.')

        except:
            print(df)
            raise(ValueError('Error trying to store the new DataFrame.'))

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
            upper:float=70.0, strat_name:str='RSIV',
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




class SecondaryIndicatorSignals:

    def __init__(self,df:pd.DataFrame=None, backtest:bool=False):

        self.df = df
        self.indicators = Indicators(df)
        self.shift = 1 if backtest else 0

    def _newDf(self, df:pd.DataFrame, errors:bool=True) -> None:

        try:
            self.df = self.df.copy() if not isinstance(df, pd.DataFrame) else df
            if 'Spread' not in df:
                if errors:
                    raise ValueError('"Spread" is not between the dataframe columns.')
                else:
                    self.df['Spread'] = [0]*len(self.df)
                    print('"Spread" is not between the dataframe columns.')
            if 'SLdist' not in df:
                if errors:
                    raise ValueError('"SLdist" is not between the dataframe columns.')
                else:
                    self.df['SLdist'] = [0]*len(self.df)
                    print('"SLdist" is not between the dataframe columns.')

        except:
            print(df)
            raise(ValueError('Error trying to store the new DataFrame.'))

    def chandeMomentum(self,df:pd.DataFrame=None, n:int=14, lower:float=-0.5,
                       upper:float=0.5, strat_name:str='ChandMOsc', exit_signal:bool=False
                       ) -> pd.DataFrame: 
        
        '''
        Buy when the Chande Momentum Oscillator crosses upwards the lower limit.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Chande Momentum Oscillator period.
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
        
        self._newDf(df)
        df = self.df.copy()
            

        df = self.indicators.chandeMomentumOscillator(n=n, dataname='CMO', 
                                                      new_df=df)

        short_condition = (df['CMO'] < upper) & \
                        (df['CMO'].shift(1) > upper)
        long_condition = (df['CMO'] > lower) & \
                        (df['CMO'].shift(1) < lower)
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

    def deMarker(self,df:pd.DataFrame=None, n:int=14, lower:float=0.2,
                       upper:float=0.8, strat_name:str='DMark', exit_signal:bool=False
                       ) -> pd.DataFrame: 
        
        '''
        Buy when the DeMarker crosses upwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            DeMarker period.
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
        
        self._newDf(df)
        df = self.df.copy()
            

        df = self.indicators.demarker(n=n, dataname='DeMark', new_df=df)

        short_condition = (df['DeMark'] > upper) & \
                        (df['DeMark'].shift(1) < upper)
        long_condition = (df['DeMark'] < lower) & \
                        (df['DeMark'].shift(1) > lower)
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

    def detrended(self,df:pd.DataFrame=None, n:int=14, lower:float=0.2,
                    upper:float=0.8, strat_name:str='DTrend', exit_signal:bool=False
                    ) -> pd.DataFrame: 
        
        '''
        Buy when the DeTrended Oscillator crosses upwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            DeTrended Oscillator period.
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
        
        self._newDf(df)
        df = self.df.copy()
            

        df = self.indicators.detrendedOscillator(n=n, method='s', datatype='Close', 
                                                 dataname='DeTrend', new_df=df)

        short_condition = (df['DeTrend'] > upper) & \
                        (df['DeTrend'].shift(1) < upper)
        long_condition = (df['DeTrend'] < lower) & \
                        (df['DeTrend'].shift(1) > lower)
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
    
    def dirProbIndex(self,df:pd.DataFrame=None, n:int=14, lower:float=0.2,
                    upper:float=0.8, strat_name:str='DProb', exit_signal:bool=False
                    ) -> pd.DataFrame: 
        
        '''
        Buy when the Directional Probability Index Oscillator crosses upwards the 
        lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Directional Probability Index Oscillator period.
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
        
        self._newDf(df)
        df = self.df.copy()
            

        df = self.indicators.directionalProbOscillator(n=n, dataname='DProbOsc', 
                                                       new_df=df)

        short_condition = (df['DProbOsc'] > upper) & \
                        (df['DProbOsc'].shift(1) < upper)
        long_condition = (df['DProbOsc'] < lower) & \
                        (df['DProbOsc'].shift(1) > lower)
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
    
    def modFisher(self,df:pd.DataFrame=None, n:int=10, lower:float=-2.0,
                    upper:float=2.0, strat_name:str='SFish', exit_signal:bool=False
                    ) -> pd.DataFrame: 
        
        '''
        Buy when the Simple Fisher Transform crosses upwards the 
        lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Simple Fisher Transform period.
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
        
        self._newDf(df)
        df = self.df.copy()
            

        df = self.indicators.simpleFisher(n=n, dataname='simpleFisher', 
                                                       new_df=df)

        short_condition = (df['simpleFisher'] > upper) & \
                        (df['simpleFisher'].shift(1) < upper)
        long_condition = (df['simpleFisher'] < lower) & \
                        (df['simpleFisher'].shift(1) > lower)
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
        
    def momentum(self,df:pd.DataFrame=None, n:int=14, lower:float=1.0,
                upper:float=1.0, strat_name:str='MOsc', exit_signal:bool=False
                ) -> pd.DataFrame: 
        
        '''
        Buy when the Momentum Oscillator crosses upwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Momentum Oscillator period.
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
        
        self._newDf(df)
        df = self.df.copy()
            

        df = self.indicators.momentumOscillator(n=n, dataname='MO', 
                                                new_df=df)  

        short_condition = (df['MO'] < upper) & \
                        (df['MO'].shift(1) > upper)
        long_condition = (df['MO'] > lower) & \
                        (df['MO'].shift(1) < lower)
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

    def paraRSI(self,df:pd.DataFrame=None, n:int=14, lower:float=20,
                upper:float=80, strat_name:str='PRSI', exit_signal:bool=False
                ) -> pd.DataFrame: 
        
        '''
        Buy when the Parabolic RSI crosses upwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Parabolic RSI period.
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
        
        self._newDf(df)
        df = self.df.copy()
            

        df = self.indicators.sar(af=0.02, amax=0.2, dataname='PSAR', new_df=df)  
        df = self.indicators.rsi(n=n, datatype='PSAR', dataname='RSI', new_df=df)  

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
    
    def relVigor(self,df:pd.DataFrame=None, n:int=14, lower:float=20,
                upper:float=80, strat_name:str='RelVigor', exit_signal:bool=False
                ) -> pd.DataFrame: 
        
        '''
        Buy when the Relative Vigor Index crosses upwards the Signal.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Parabolic RSI period.
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
        
        self._newDf(df)
        df = self.df.copy()
            

        df = self.indicators.relativeVigorOscillator(n=n, datatype='RVI', dataname='Close', new_df=df)  

        short_condition = (df['RVI'] < df['RVISig']) & \
                        (df['RVI'].shift(1) > df['RVISig'].shift(1)) & \
                        (df['RVI'] > 0)
        long_condition = (df['RVI'] > df['RVISig']) & \
                        (df['RVI'].shift(1) < df['RVISig'].shift(1)) & \
                        (df['RVI'] < 0)
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

    def rsiAtr(self,df:pd.DataFrame=None, n:int=14, m:int=14, o:int=14, lower:float=30,
                upper:float=70, strat_name:str='RsiAtr', exit_signal:bool=False
                ) -> pd.DataFrame: 
        
        '''
        Buy when the RSIATR crosses upwards the Signal.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            ATR period.
        m: int
            First RSI period.
        o: int
            Second RSI period.
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
        
        self._newDf(df)
        df = self.df.copy()
            

        df = self.indicators.rsiAtr(n=n, m=m, o=o, datatype='Close', dataname='RSIATR', new_df=df)  

        short_condition = (df['RSIATR'] > upper) & \
                        (df['RSIATR'].shift(1) < upper)
        long_condition = (df['RSIATR'] < lower) & \
                        (df['RSIATR'].shift(1) > lower)
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

    def stochRsi(self,df:pd.DataFrame=None, n:int=14, m:int=3, p:int=3, lower:float=5,
                upper:float=95, strat_name:str='StochRsi', exit_signal:bool=False
                ) -> pd.DataFrame: 
        
        '''
        Buy when the Stochstic RSI crosses upwards the Signal.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Stochstic and RSI period.
        m: int
            Length of the slow moving average.
        p: int
            Length of the fast moving average.
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
        
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.rsi(n=n, method='s',datatype='Close',dataname='RSI',new_df=df)
        df = self.indicators.stochasticOscillator(n=n, m=m, p=p, datatype='RSI', dataname='SO', new_df=df)  

        short_condition = (df['SO'] > upper) & \
                        (df['SO'].shift(1) < upper)
        long_condition = (df['SO'] < lower) & \
                        (df['SO'].shift(1) > lower)
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


class KSignals:

    def __init__(self,df:pd.DataFrame=None, backtest:bool=False):

        self.df = df
        self.indicators = Indicators(df)
        self.shift = 1 if backtest else 0

    def _newDf(self, df:pd.DataFrame, errors:bool=True) -> None:

        try:
            self.df = self.df.copy() if not isinstance(df, pd.DataFrame) else df
            if 'Spread' not in df:
                if errors:
                    raise ValueError('"Spread" is not between the dataframe columns.')
                else:
                    self.df['Spread'] = [0]*len(self.df)
                    print('"Spread" is not between the dataframe columns.')
            if 'SLdist' not in df:
                if errors:
                    raise ValueError('"SLdist" is not between the dataframe columns.')
                else:
                    self.df['SLdist'] = [0]*len(self.df)
                    print('"SLdist" is not between the dataframe columns.')

        except:
            print(df)
            raise(ValueError('Error trying to store the new DataFrame.'))
        
    def envelopes(self,df:pd.DataFrame=None, n:int=14, strat_name:str='EnvStrat', 
                 exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Low drops between the highs and lows MAs and the Close is 
        below the higher MA.


        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            ATR period.
        m: int
            Length of the slow moving average.
        p: int
            Length of the fast moving average.
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
        
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.movingAverage(n=n, method='s',datatype='High',dataname='UPMA',new_df=df)
        df = self.indicators.movingAverage(n=n, method='s',datatype='Low',dataname='DNMA',new_df=df)

        short_condition = (df['High'] < df['UPMA']) & (df['High'] > df['DNMA']) & \
                        (df['High'].shift(1) > df['DNMA'].shift(1)) & \
                        (df['Close'] > df['DNMA'])
        long_condition = (df['Low'] < df['UPMA']) & (df['Low'] > df['DNMA']) & \
                        (df['Low'].shift(1) > df['UPMA'].shift(1)) & \
                        (df['Close'] < df['UPMA'])
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
     
    def fibEnvelopes(self,df:pd.DataFrame=None, n:int=14, strat_name:str='FibEnv', 
                 exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Low drops between the highs and lows FMAs and the Close is 
        below the higher FMA.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            ATR period.
        m: int
            Length of the slow moving average.
        p: int
            Length of the fast moving average.
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
        
        self._newDf(df)
        df = self.df.copy()
            
        df = self.indicators.movingAverage(n=n, method='f',datatype='High',dataname='UPMA',new_df=df)
        df = self.indicators.movingAverage(n=n, method='f',datatype='Low',dataname='DNMA',new_df=df)

        short_condition = (df['High'] < df['UPMA']) & (df['High'] > df['DNMA']) & \
                        (df['High'].shift(1) > df['DNMA'].shift(1)) & \
                        (df['Close'] > df['DNMA'])
        long_condition = (df['Low'] < df['UPMA']) & (df['Low'] > df['DNMA']) & \
                        (df['Low'].shift(1) > df['UPMA'].shift(1)) & \
                        (df['Close'] < df['UPMA'])
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

    def fibTiming(self,df:pd.DataFrame=None, count:int=8, n:int=5, m:int=3, 
                  strat_name:str='FibEnv', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Fibonacci pattern appears.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        count: int
            Counter for the pattern.
        n: int
            Step one for the pattern.
        m: int
            Step two for the pattern.
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
        
        self._newDf(df)
        df = self.df.copy()
            
        counter = -1
        long = []
        for i,idx in enumerate(min([n,m]),df.index):
            candle = df.iloc[i]
            candle_one = df.iloc[i-n]
            candle_two = df.iloc[i-m]
            if candle['Close'] < candle_one['Close'] and \
                candle['Close'] < candle_two['Close']:
                
                long.append(counter)
                counter += -1   
                
                if counter == -count - 1:
                    counter = 0
                else:
                    continue  
                
            elif candle['Close'] >= candle_one['Close'] or \
                candle['Close'] >= candle_two['Close']:
                
                counter = -1
                long.append(0) 
            
        counter = 1 
        short = []
        for i,idx in enumerate(min([n,m]),df.index):
            candle = df.iloc[i]
            candle_one = df.iloc[i-n]
            candle_two = df.iloc[i-m]
            if candle['Close'] > candle_one['Close'] and \
                candle['Close'] > candle_two['Close']:
                
                short.append(counter) 
                counter += 1      
                
                if counter == count + 1: 
                    counter = 0     
                else:
                    continue        
                
            elif candle['Close'] <= candle_one['Close'] or \
                candle['Close'] <= candle_two['Close']:
                
                counter = 1 
                short.append(0) 
        
        df['Long'] = long
        df['Short'] = short


        short_condition = (df['Short'] == count) & (df['Close'] > df['Close'].shift(1))
        long_condition = (df['Long'] == -count) & (df['Close'] < df['Close'].shift(1))
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

    def supRes(self,df:pd.DataFrame=None, n:int=5, lower:float=0.05, upper:float=0.05,
                  strat_name:str='SupRes', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Fibonacci pattern appears.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        count: int
            Counter for the pattern.
        n: int
            Step one for the pattern.
        m: int
            Step two for the pattern.
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
        
        self._newDf(df)
        df = self.df.copy()
        
        df['Range'] = df['High'].rolling(n).max() - df['Low'].rolling(n).min()
        df['Support'] = df['Low'].rolling(n).min() + df['Close']*lower
        df['Resistance'] = df['High'].rolling(n).max() - df['Close']*upper


        short_condition = (df['Close'] < df['Resistance']) & \
                        (df['Close'].shift(1) > df['Resistance'].shift(1))
        long_condition = (df['Close'] > df['Support']) & \
                        (df['Close'].shift(1) < df['Support'].shift(1))
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
    
    def reversal(self,df:pd.DataFrame=None, n:int=100, dev:float=2,
                 strat_name:str='RevStrat', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Fibonacci pattern appears.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        count: int
            Counter for the pattern.
        n: int
            Step one for the pattern.
        m: int
            Step two for the pattern.
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
        
        self._newDf(df)
        df = self.df.copy()
        
        df = self.indicators.bollingerBands(n=n, method='s', desvi=dev, datatype='Close', 
                                            dataname='BB', new_df=df)
        df = self.indicators.macd(a=12, b=26, c=9, datatype='Close', dataname='MACD', new_df=df)  


        short_condition = ((df['Close'] > df['BBUP']) | (df['Open'] > df['BBUP'])) & \
                        (df['MACD'] < df['MACDS']) & (df['MACD'].shift(1) > df['MACDS'].shift(1))
        long_condition = ((df['Close'] < df['BBDN']) | (df['Open'] < df['BBDN'])) & \
                        (df['MACD'] > df['MACDS']) & (df['MACD'].shift(1) < df['MACDS'].shift(1))
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

    def _newDf(self, df:pd.DataFrame, errors:bool=True) -> None:

        try:
            self.df = self.df.copy() if not isinstance(df, pd.DataFrame) else df
            if 'Spread' not in df:
                if errors:
                    raise ValueError('"Spread" is not between the dataframe columns.')
                else:
                    self.df['Spread'] = [0]*len(self.df)
                    print('"Spread" is not between the dataframe columns.')
            if 'SLdist' not in df:
                if errors:
                    raise ValueError('"SLdist" is not between the dataframe columns.')
                else:
                    self.df['SLdist'] = [0]*len(self.df)
                    print('"SLdist" is not between the dataframe columns.')

        except:
            print(df)
            raise(ValueError('Error trying to store the new DataFrame.'))
    
    def turtlesBreakout(self, df:pd.DataFrame=None, 
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

        
    if False:
        from degiro import DeGiro
        degiro = DeGiro('OneMade','Onemade3680')
        products = degiro.getProducts(exchange_id=663,country=846) # Nasdaq exchange
        asset = products.iloc[213] # AAPL -> vwdid = 350015372
        raw = degiro.getPriceData(asset['vwdId'], 'PT1H', 'P5Y', tz='UTC')
    else:
        import yfinance as yf
        raw = yf.Ticker('SPY').history(period='max',interval='1h')

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
