
import numpy as np
import pandas as pd

from indicators import OHLC, Indicators

    
# def checkStrategyConfig(self, strat_name):

#     from config import strategies

#     if strat_name not in list(strategies.keys()):
#         raise ValueError(f'Strategy "{strat_name}" not between the tradeable ' + \
#                             'list: '+','.join(list(strategies.keys())))

class SignalsTemplate(OHLC):

    class Side:
        ALL = 0
        LONG = 1
        SHORT = 2

    needed_cols = ['Open', 'High', 'Low', 'Close', 'Spread', 'SLdist']

    def __init__(self,df:pd.DataFrame=None, backtest:bool=False, side:Side=Side.ALL, 
                 errors:bool=True, verbose:bool=True) -> None:

        self.errors = errors
        self.verbose = verbose
        self.side = side
        self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        self.indicators = Indicators(df, errors=errors, verbose=verbose)
        self.shift = 1 if backtest else 0

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
        
    def _renameEntry(self, strat:str) -> str:

        return f'{strat}Entry'
    
    def _renameExit(self, strat:str) -> str:

        return f'{strat}Exit'

    def _generateSignal(self, df:pd.DataFrame, strat:str, long_condition:pd.Series, 
                        short_condition:pd.Series, exe_condition:pd.DataFrame
                        ) -> pd.DataFrame:
        
        df[self._renameEntry(strat)] = np.where(exe_condition.shift(self.shift) & \
                        long_condition.shift(self.shift), (1 if self.side != self.Side.SHORT else 0),
                    np.where(exe_condition.shift(self.shift) & \
                        short_condition.shift(self.shift), (-1 if self.side != self.Side.LONG else 0), 
                    0))

        return df

class OscillatorSignals(SignalsTemplate):
    
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
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = getattr(self.indicators, ind_name)(**kwargs)
        
        ind = kwargs['dataname']
        short_condition = (df[ind] > upper) & (df[ind].shift(1) < upper)
        long_condition = (df[ind] < lower) & (df[ind].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = getattr(self.indicators, ind_name)(**kwargs)

        ind = kwargs['dataname']
        short_condition = (df[ind] < upper) & (df[ind].shift(1) > upper)
        long_condition = (df[ind] > lower) & (df[ind].shift(1) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

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

        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
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
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = getattr(self.indicators, ind_name)(**kwargs)

        ind = kwargs['dataname']

        short_condition = (df[ind] > upper) & \
                        (df[ind].shift(1) < upper)
        long_condition = (df[ind] < lower) & \
                        (df[ind].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
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
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

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

        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

class PrimaryIndicatorSignals(SignalsTemplate):

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.bollingerBands(n=n, method='s', desvi=dev, 
                                            datatype='Close', dataname='BB', 
                                            new_df=df)

        short_condition = (df['Close'] > df['BBUP']) & \
                        (df['Close'].shift(1) < df['BBUP'].shift(1))
        long_condition = (df['Close'] < df['BBDN']) & \
                        (df['Close'].shift(1) > df['BBDN'].shift(1))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.bollingerBands(n=n, method='s', desvi=dev, 
                                            datatype='Close', dataname='BB', 
                                            new_df=df)

        short_condition = (df['Close'] < df['BBUP']) & \
                        (df['Close'].shift(1) > df['BBUP'].shift(1)) & \
                        (df['Close'] > df['BBDN']+df['BBW']/2)
        long_condition = (df['Close'] > df['BBDN']) & \
                        (df['Close'].shift(1) < df['BBDN'].shift(1)) & \
                        (df['Close'] < df['BBDN']+df['BBW']/2)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.pctBollinger(n=n, desvi=dev, datatype='Close', 
                                          dataname='PctBB', new_df=df)
        
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
        
        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                 dataname='RSI', new_df=df)

        short_condition = (df['RSI'] > upper) & (df['RSI'].shift(1) < upper)
        long_condition = (df['RSI'] < lower) & (df['RSI'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                 dataname='RSI', new_df=df)

        short_condition = (df['RSI'] < upper) & (df['RSI'].shift(1) > upper)
        long_condition = (df['RSI'] > lower) & (df['RSI'].shift(1) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.movingAverage(n=m, method='s', datatype='Close', 
                                           dataname='MA', new_df=df)
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                 dataname='RSI', new_df=df)

        short_condition = (df['RSI'] > upper) & \
                        (df['Close'].shift(1) > df['MA'].shift(1)) & \
                        (df['Close'] < df['MA'])
        long_condition = (df['RSI'] < lower) & \
                        (df['Close'].shift(1) < df['MA'].shift(1)) & \
                        (df['Close'] > df['MA'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                 dataname='RSI', new_df=df)

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

        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                 dataname='RSI', new_df=df)

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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                 dataname='RSI', new_df=df)

        short_condition = (df['RSI'] > upper) & \
                        (df['RSI'].shift(1) < upper)
        long_condition = (df['RSI'] < lower) & \
                        (df['RSI'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                 dataname='RSI', new_df=df)

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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                 dataname='RSI', new_df=df)

        short_condition = (df['RSI'] <= upper) & \
                        (df['RSI'].shift(1) >= upper-tolerance) & \
                        (df['RSI'].shift(2) <= df['RSI'])
        long_condition = (df['RSI'] >= lower) & \
                        (df['RSI'].shift(1) <= lower+tolerance) & \
                        (df['RSI'].shift(2) >= df['RSI'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                 dataname='RSI', new_df=df)

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

        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                 dataname='RSI', new_df=df)

        short_condition = (df['RSI'] < upper) & \
                        (df['RSI'].shift(1) > upper) & \
                        (df['RSI'].shift(2) < upper)
        long_condition = (df['RSI'] > lower) & \
                        (df['RSI'].shift(1) < lower) & \
                        (df['RSI'].shift(2) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
        
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, method='s', 
                                                  dataname='SO', new_df=df)

        short_condition = (df['SOK'] > upper) & (df['SOK'].shift(1) < upper)
        long_condition = (df['SOK'] < lower) & (df['SOK'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
        
        df = self.indicators.stochasticOscillator(n=n, dataname='SO', new_df=df)

        short_condition = (df['SOK'] < upper) & (df['SOK'].shift(1) > upper)
        long_condition = (df['SOK'] > lower) & (df['SOK'].shift(1) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
        
        df = self.indicators.stochasticOscillator(n=n, m=m, p=3, dataname='SO', 
                                                  new_df=df)

        short_condition = (df['SOD'] < df['SOK']) & \
                        (df['SOD'].shift(1) > df['SOK'].shift(1)) & \
                        (df['SOD'] > upper)
        long_condition = (df['SOD'] > df['SOK']) & \
                        (df['SOD'].shift(1) < df['SOK'].shift(1)) & \
                        (df['SOD'] < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, dataname='SO', 
                                                  new_df=df)

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

        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, dataname='SO', 
                                                  new_df=df)

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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, dataname='SO', 
                                                  new_df=df)

        short_condition = (df['SOD'] > upper) & \
                        (df['SOD'].shift(1) < upper)
        long_condition = (df['SOD'] < lower) & \
                        (df['SOD'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, dataname='SO', 
                                                  new_df=df)

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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, dataname='SO', 
                                                  new_df=df)

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

        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

class SecondaryIndicatorSignals(SignalsTemplate):

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            

        df = self.indicators.chandeMomentumOscillator(n=n, dataname='CMO', 
                                                      new_df=df)

        short_condition = (df['CMO'] < upper) & \
                        (df['CMO'].shift(1) > upper)
        long_condition = (df['CMO'] > lower) & \
                        (df['CMO'].shift(1) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            

        df = self.indicators.demarker(n=n, dataname='DeMark', new_df=df)

        short_condition = (df['DeMark'] > upper) & \
                        (df['DeMark'].shift(1) < upper)
        long_condition = (df['DeMark'] < lower) & \
                        (df['DeMark'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            

        df = self.indicators.detrendedOscillator(n=n, method='s', datatype='Close', 
                                                 dataname='DeTrend', new_df=df)

        short_condition = (df['DeTrend'] > upper) & \
                        (df['DeTrend'].shift(1) < upper)
        long_condition = (df['DeTrend'] < lower) & \
                        (df['DeTrend'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            

        df = self.indicators.directionalProbOscillator(n=n, dataname='DProbOsc', 
                                                       new_df=df)

        short_condition = (df['DProbOsc'] > upper) & \
                        (df['DProbOsc'].shift(1) < upper)
        long_condition = (df['DProbOsc'] < lower) & \
                        (df['DProbOsc'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))
        self.ohlc_df = df

        return self.ohlc_df
    
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            

        df = self.indicators.simpleFisher(n=n, dataname='simpleFisher', 
                                                       new_df=df)

        short_condition = (df['simpleFisher'] > upper) & \
                        (df['simpleFisher'].shift(1) < upper)
        long_condition = (df['simpleFisher'] < lower) & \
                        (df['simpleFisher'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
        
    def momentum(self,df:pd.DataFrame=None, n:int=14, lower:float=100.0,
                upper:float=100.0, strat_name:str='MOsc', exit_signal:bool=False
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            

        df = self.indicators.momentumOscillator(n=n, dataname='MO', 
                                                new_df=df)  

        short_condition = (df['MO'] < upper) & \
                        (df['MO'].shift(1) > upper)
        long_condition = (df['MO'] > lower) & \
                        (df['MO'].shift(1) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            

        df = self.indicators.sar(af=0.02, amax=0.2, dataname='PSAR', new_df=df)  
        df = self.indicators.rsi(n=n, datatype='PSAR', dataname='RSI', new_df=df)  

        short_condition = (df['RSI'] > upper) & \
                        (df['RSI'].shift(1) < upper)
        long_condition = (df['RSI'] < lower) & \
                        (df['RSI'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def relVigor(self,df:pd.DataFrame=None, n:int=14, 
                 strat_name:str='RelVigor', exit_signal:bool=False
                ) -> pd.DataFrame: 
        
        '''
        Buy when the Relative Vigor Index crosses upwards the Signal.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Parabolic RSI period.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            

        df = self.indicators.relativeVigorOscillator(n=n, method='s', dataname='RVI', new_df=df)  

        short_condition = (df['RVI'] < df['RVISig']) & \
                        (df['RVI'].shift(1) > df['RVISig'].shift(1)) & \
                        (df['RVI'] > 0)
        long_condition = (df['RVI'] > df['RVISig']) & \
                        (df['RVI'].shift(1) < df['RVISig'].shift(1)) & \
                        (df['RVI'] < 0)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))
            
        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            

        df = self.indicators.rsiAtr(n=n, m=m, o=o, datatype='Close', dataname='RSIATR', new_df=df)  

        short_condition = (df['RSIATR'] > upper) & \
                        (df['RSIATR'].shift(1) < upper)
        long_condition = (df['RSIATR'] < lower) & \
                        (df['RSIATR'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.rsi(n=n, method='e',datatype='Close',dataname='RSI',new_df=df)
        df = self.indicators.stochasticOscillator(n=n, m=m, p=p, datatype='RSI', dataname='SO', new_df=df)  

        short_condition = (df['SOD'] > upper) & \
                        (df['SOD'].shift(1) < upper)
        long_condition = (df['SOD'] < lower) & \
                        (df['SOD'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

class KSignals(SignalsTemplate):

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
     
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        counter = -1
        long = []
        for i,idx in enumerate(df.index):
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
        for i,idx in enumerate(df.index):
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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

    def supRes(self,df:pd.DataFrame=None, n:int=5, lower:float=0.05, upper:float=0.05,
                  strat_name:str='SupRes', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when price crosses upwards the support.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Lookback for supports and resistances.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
        
        df['Range'] = df['High'].rolling(n).max() - df['Low'].rolling(n).min()
        df['Support'] = df['Low'].rolling(n).min() + df['Close']*lower
        df['Resistance'] = df['High'].rolling(n).max() - df['Close']*upper


        short_condition = (df['Close'] < df['Resistance']) & \
                        (df['Close'].shift(1) > df['Resistance'].shift(1))
        long_condition = (df['Close'] > df['Support']) & \
                        (df['Close'].shift(1) < df['Support'].shift(1))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def reversal(self,df:pd.DataFrame=None, n:int=100, dev:float=2,
                 strat_name:str='RevStrat', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when Price closes below lower Bollinger Band and MACD crosses upwards.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Lookback for the Bollinger Bands and MACD.
        dev: float
            Deviation for the Bollinger Bands.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
        
        df = self.indicators.bollingerBands(n=n, method='s', desvi=dev, datatype='Close', 
                                            dataname='BB', new_df=df)
        df = self.indicators.macd(a=12, b=26, c=9, datatype='Close', dataname='MACD', new_df=df)  


        short_condition = ((df['Close'] > df['BBUP']) | (df['Open'] > df['BBUP'])) & \
                        (df['MACD'] < df['MACDS']) & (df['MACD'].shift(1) > df['MACDS'].shift(1))
        long_condition = ((df['Close'] < df['BBDN']) | (df['Open'] < df['BBDN'])) & \
                        (df['MACD'] > df['MACDS']) & (df['MACD'].shift(1) < df['MACDS'].shift(1))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def volatBands(self,df:pd.DataFrame=None, n:int=20, mult:float=2,
                 strat_name:str='VolatB', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when Close crosses downwards the lower Volatility Band.

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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
        
        df = self.indicators.volatilityBands(n=n, multiplier=mult, datatype='Close', 
                                            dataname='VB', new_df=df)


        short_condition = (df['Close'] > df['VBUP']) & (df['Close'].shift(1) < df['VBUP'].shift(1))
        long_condition = (df['Close'] < df['VBDN']) & (df['Close'].shift(1) > df['VBDN'].shift(1))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

class ContrarianStrategies(SignalsTemplate):
        
    def macdStrat(self,df:pd.DataFrame=None, a:int=12, b:int=26, c:int=9, 
                  width:int=60, strat_name:str='MACDStrat', 
                  exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Low drops between the highs and lows MAs and the Close is 
        below the higher MA.


        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        a: int
            MACD length of the fast moving average.
        b: int
            MACD length of the slow moving average.
        c: int
            MACD length of the difference moving average.
        width: int
            Amplitude for the divergence.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.macd(a=a, b=b, c=c, method='e', datatype='Close', 
                                  dataname='MACD', new_df=df)

        max_i = len(df)
        long_condition = []
        short_condition = []
        for i,idx  in enumerate(df.index):
            candle = df.iloc[i]
            done = False

            # Long
            if candle['MACD'] < candle['MACDS'] and candle['MACD'] < 0:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    high = df.iloc[j]
                    if high['MACD'] > high['MACDS'] and high['MACD'] < 0:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            higher_low = df.iloc[k]
                            if higher_low['MACD'] < higher_low['MACDS'] and \
                                higher_low['MACD'] < 0 and \
                                higher_low['Close'] < candle['Close'] and \
                                higher_low['MACD'] > candle['MACD']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if df.iloc[l]['MACDS'] < df.iloc[l]['MACD'] and \
                                        df.iloc[l]['MACD'] < 0:
                                        long_condition.append(True)
                                        short_condition.append(False)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break

            # Short
            elif candle['MACD'] > candle['MACDS'] and candle['MACD'] > 0:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    low = df.iloc[j]
                    if low['MACD'] < low['MACDS'] and low['MACD'] > 0:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            lower_high = df.iloc[k]
                            if lower_high['MACD'] > lower_high['MACDS'] and \
                                lower_high['MACD'] > 0 and \
                                lower_high['Close'] > candle['Close'] and \
                                lower_high['MACD'] < candle['MACD']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if df.iloc[l]['MACDS'] > df.iloc[l]['MACD'] and \
                                        df.iloc[l]['MACD'] > 0:
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

        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
        
    def timingStrat(self,df:pd.DataFrame=None, n:int=1, lower:float=-5, 
                  upper:float=5, strat_name:str='MACDStrat', 
                  exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the timing indicator crosses the lower level upwards.


        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Timing indicator length.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df['Diff'] = df['Close'] - df['Close'].shift(n)
        upt = [0]
        dnt = [0]
        for i in df.index:
            if df.loc[i]['Diff'] > 0:
                upt.append(upt[-1] + 1)
                dnt.append(0)
            elif df.loc[i]['Diff'] < 0:
                upt.append(0)
                dnt.append(dnt[-1] - 1)
            else:
                upt.append(0)
                dnt.append(0)
        df['UPT'] = upt[-len(df):]
        df['DNT'] = dnt[-len(df):]
        df['Timing'] = df['UPT'] + df['DNT']

        short_condition = (df['Timing'] <= upper) & (df['Timing'].shift(1) > upper) & \
                        (df['Timing'].shift(2) <= upper)
        long_condition = (df['Timing'] >= lower) & (df['Timing'].shift(1) < lower) & \
                        (df['Timing'].shift(2) >= lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def fisherRsiStrat(self,df:pd.DataFrame=None, n:int=14, m:int=5, 
                       lower:float=15, upper:float=85, strat_name:str='FRSIStrat', 
                       exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the timing indicator crosses the lower level downwards.


        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Fisher Transform indicator length.
        m: int
            RSI indicator length.
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

        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.simpleFisher(n=n, m=3, p=3, method='s', dataname='SFisher',
                                          new_df=df)
        df = self.indicators.rsi(n=m, method='e', datatype='SFisher', dataname='RSI', 
                                 new_df=df)

        short_condition = (df['RSI'] > upper) & (df['RSI'].shift(1) < upper)
        long_condition = (df['RSI'] < lower) & (df['RSI'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def vamaStochStrat(self,df:pd.DataFrame=None, n:int=20, m:int=100, p:int=30, 
                       lower:float=20, upper:float=80, strat_name:str='VAMAStrat', 
                       exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Stochastic is oversold and the price crosses the lower VAMA 
        band upwards.


        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            VAMA bands and Stochastic Oscillator lengths.
        m: int
            Length of the larger volatility for adjusting of the VAMA bands.
        p: int
            Length for the volatility of the VAMA bands.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.vamaBands(n=n, m=m, p=p, desvi=3, datatype='Close', dataname='VB',
                                        new_df=df)
        df = self.indicators.stochasticOscillator(n=n, m=1, p=2, method='s', dataname='SO',
                                        new_df=df)

        short_condition = (df['High'] > df['VBUP'].shift(1)) & \
                        (df['VBUP'].shift(1) > df['High'].shift(1)) & \
                        (df['SOK'] > upper)
        long_condition = (df['Low'] < df['VBDN'].shift(1)) & \
                        (df['VBDN'].shift(1) > df['Low'].shift(1)) & \
                        (df['SOK'] < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def rsiDivStrat(self,df:pd.DataFrame=None, n:int=14, m:int=5, width:int=60,
                       lower:float=20, upper:float=80, strat_name:str='RSIDivStrat', 
                       exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI of the RSI makes a divergence with price.


        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the first RSI.
        m: int
            Length of the second RSI.
        width: int
            Divergence width.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.rsi(n=n, method='e', datatype='Close', dataname='RSIF', 
                                 new_df=df)
        df = self.indicators.rsi(n=m, method='e', datatype='RSIF', dataname='RSI', 
                                 new_df=df)

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

        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

    def keltnerStrat(self,df:pd.DataFrame=None, n:int=10, strat_name:str='VAMAStrat', 
                       exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when theprice crosses the lower Keltner Channel line upwards.


        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Keltner Channel length.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.keltnerChannel(n=n, mamethod='s', atrn=n, atrmethod='s', multiplier=2,
                                        datatype='Close', dataname='KC', new_df=df)

        short_condition = (df['High'] > df['KCUP'].shift(1)) & \
                        (df['KCUP'].shift(1) > df['High'].shift(1))
        long_condition = (df['Low'] < df['KCDN'].shift(1)) & \
                        (df['KCDN'].shift(1) > df['Low'].shift(1))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def slopeStrat(self,df:pd.DataFrame=None, n:int=14, m:int=14, 
                       lower:float=30, upper:float=70, strat_name:str='SlopeStrat', 
                       exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI of the slope crosses the lower level upwards.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Slope indicator length.
        m: int
            RSI indicator length.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.slope(n=n, dataname='Slope', pct=False, new_df=df)
        df = self.indicators.rsi(n=m, method='e', datatype='Slope', dataname='RSI', 
                                 new_df=df)

        short_condition = (df['RSI'] > upper) & (df['RSI'].shift(1) < upper)
        long_condition = (df['RSI'] < lower) & (df['RSI'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def robBookerStrat(self,df:pd.DataFrame=None, a:int=12, b:int=26, c:int=9, n:int=70, 
                       lower:float=30, upper:float=70, strat_name:str='RobBookerRev', 
                       exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the MACD crosses the 0 level upwards and the Stochastic crosses the lower 
        level downwards.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        a: int
            MACD fast moving average length.
        b: int
            MACD slow moving average length.
        c: int
            MACD signal length.
        n: int
            Stochastic indicator length.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.macd(a=a, b=b, c=c, method='e', datatype='Close', 
                                  dataname='MACD', new_df=df)
        df = self.indicators.stochasticOscillator(n=n, m=10, p=10, method='s', dataname='SO',
                                    new_df=df)

        short_condition = (df['SOD'] > upper) & (df['MACD'].shift(1) > 0) & (0 > df['MACD'])
        long_condition = (df['SOD'] < lower) & (df['MACD'].shift(1) < 0) & (0 < df['MACD'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def rsiMStrat(self,df:pd.DataFrame=None, n:int=14, width:int=40, 
                       lower:float=30, upper:float=70, strat_name:str='RSIM', 
                       exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI creates a double top pattern.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            RSI indicator length.
        width: int
            Pattern width.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.rsi(n=n, method='e', datatype='Close', dataname='RSI', 
                                 new_df=df)

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
                    if lower < high['RSI']:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            higher_low = df.iloc[k]
                            if higher_low['RSI'] < lower and \
                                higher_low['RSI'] >= candle['RSI']:
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
                    if low['RSI'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            lower_high = df.iloc[k]
                            if lower_high['RSI'] > upper and \
                                lower_high['RSI'] <= candle['RSI']:
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

        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def mandiStrat(self,df:pd.DataFrame=None, n:int=20, m:int=100,
                    strat_name:str='mandiStrat', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the indicator equals 100 and the price is between the fast and low 
        moving averages.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Index moving average.
        m: int
            Length of the trend moving average.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.movingAverage(n=n, method='s', datatype='Close', 
                                        dataname='FMA', new_df=df)
        df = self.indicators.movingAverage(n=m, method='s', datatype='Close', 
                                        dataname='SMA', new_df=df)
        df['NI'] = abs(df['Close'] - df['FMA'])
        df['MANDI'] = (df['NI'] - df['NI'].rolling(n).min()) / \
                    (df['NI'].rolling(n).max() - df['NI'].rolling(n).min())

        short_condition = (df['MANDI'] == 1) & (df['MANDI'].shift(1) < 1) & \
                        (df['Close'] < df['SMA']) & (df['Close'] > df['FMA'])
        long_condition = (df['MANDI'] == 1) & (df['MANDI'].shift(1) < 1) & \
                        (df['Close'] > df['SMA']) & (df['Close'] < df['FMA'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def bbRsiStrat(self,df:pd.DataFrame=None, n:int=14, m:int=20, std:float=2, 
                       lower:float=30, upper:float=70, strat_name:str='BBrevStrat', 
                       exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the price is below the bollinger band and the RSI crosses 
        downwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the RSI.
        m: int
            Length of the Bollinger Bands.
        std: float
            Standard deviation of the Bollinger Bands.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                dataname='RSI', new_df=df)
        df = self.indicators.bollingerBands(n=m, method='s', desvi=std, 
                                datatype='Close', dataname='BB', new_df=df)
        
        short_condition = (df['Close'] > df['BBUP']) & (df['RSI'] > upper) & \
                        (df['RSI'].shift(1) < upper)
        long_condition = (df['Close'] < df['BBDN']) & (df['RSI'] < lower) & \
                        (df['RSI'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def stochRsiStrat(self,df:pd.DataFrame=None, n:int=14, m:int=5, 
                       lower:float=30, upper:float=70, strat_name:str='OSoldStrat', 
                       exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when both the RSI and Stochastic indicators cross upwards the lower level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the RSI.
        m: int
            Length of the Stochastic Oscillator.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                dataname='RSI', new_df=df)
        df = self.indicators.stochasticOscillator(n=m, m=3, p=3, method='s', 
                                dataname='SO', new_df=df)
        
        short_condition = (df['SOK'] < upper) & (df['SOK'].shift(1) > upper) & \
                        (df['RSI'] < upper) & (df['RSI'].shift(1) > upper)
        long_condition = (df['SOK'] > lower) & (df['SOK'].shift(1) < lower) & \
                        (df['RSI'] > lower) & (df['RSI'].shift(1) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
class TrendStrategies(SignalsTemplate):

    def maCross(self,df:pd.DataFrame=None, n:int=50, m:int=100, strat_name:str='MaCross', 
               exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the fast moving average crosses the slow one upwards.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the fast Moving Average.
        m: int
            Length of the slow Moving Average.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.movingAverage(n=n, method='s', datatype='Close', 
                                        dataname='FMA', new_df=df)
        df = self.indicators.movingAverage(n=m, method='s', datatype='Close', 
                                        dataname='SMA', new_df=df)
        
        short_condition = (df['SMA'].shift(1) < df['FMA'].shift(1)) & (df['SMA'] > df['FMA'])
        long_condition = (df['SMA'].shift(1) > df['FMA'].shift(1)) & (df['SMA'] < df['FMA'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df

    def maZone(self,df:pd.DataFrame=None, n:int=100, strat_name:str='MaZone', 
               exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when price is between the moving averages of the Highs and the Lows after 
        being above the Highs moving average the period before and 10 times before.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Averages.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.movingAverage(n=n, method='s', datatype='High', 
                                        dataname='HMA', new_df=df)
        df = self.indicators.movingAverage(n=n, method='s', datatype='Low', 
                                        dataname='LMA', new_df=df)
        
        short_condition = (df['High'] < df['HMA']) & (df['High'] > df['LMA']) & \
                        (df['High'].shift(1) < df['LMA']) & (df['High'].shift(10) < df['LMA'])
        long_condition = (df['Low'] < df['HMA']) & (df['Low'] > df['LMA']) & \
                        (df['Low'].shift(1) > df['HMA']) & (df['Low'].shift(10) > df['HMA'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def maRsi(self,df:pd.DataFrame=None, n:int=50, m:int=2, lower:float=5, upper:float=95, 
              strat_name:str='MaRSITrend', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when price is above the moving average and the RSI crosses the lower level upwards.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Average.
        m: int
            Length of the RSI.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.movingAverage(n=n, method='s', datatype='Close', 
                                        dataname='MA', new_df=df)
        df = self.indicators.rsi(n=m, method='e', datatype='Close', 
                                        dataname='RSI', new_df=df)
        
        short_condition = (df['Close'] < df['MA']) & (df['RSI'] > upper) & \
                        (df['RSI'].shift(1) < upper)
        long_condition = (df['Close'] > df['MA']) & (df['RSI'] < lower) & \
                        (df['RSI'].shift(1) > lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def maSlope(self,df:pd.DataFrame=None, n:int=20, m:int=20, lower:float=0, upper:float=0, 
              strat_name:str='MaSlopTrend', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the slope of the moving average changes its sign.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Average.
        m: int
            Length of the Slope.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.movingAverage(n=n, method='s', datatype='Close', 
                                           dataname='MA', new_df=df)
        df = self.indicators.slope(n=m, datatype='MA', dataname='Slope', pct=False, 
                                   new_df=df)
        
        short_condition = (df['Slope'] < upper) & (df['Slope'].shift(1) > upper)
        long_condition = (df['Slope'] > lower) & (df['Slope'].shift(1) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def macdTrend(self,df:pd.DataFrame=None, a:int=12, b:int=26, c:int=9, lower:float=0, 
                  upper:float=0, strat_name:str='MacdTrend', exit_signal:bool=False
                  ) -> pd.DataFrame: 
        
        '''
        Buy when MACD signal crosses above the lower level (0).

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        a: int
            Length of the fast moving average.
        b: int
            Length of the slow moving average.
        c: int
            Length of the difference moving average.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.macd(a=a, b=b, c=c, method='s', datatype='Close', 
                                        dataname='MACD', new_df=df)
        
        short_condition = (df['MACDS'] < upper) & (df['MACDS'].shift(1) > upper)
        long_condition = (df['MACDS'] > lower) & (df['MACDS'].shift(1) < lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def superTrend(self,df:pd.DataFrame=None, n:int=10, m:int=4,
              strat_name:str='SuperTrend', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the price crosses upwards the SuperTrend indicator.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the SuperTrend.
        m: int
            ATR multiplier for the SuperTrend.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.superTrend(n=n, method='s', mult=m, datatype='Close', 
                                           dataname='ST', new_df=df)
        
        short_condition = (df['Close'] < df['ST']) & (df['Close'].shift(1) > df['ST'].shift(1))
        long_condition = (df['Close'] > df['ST']) & (df['Close'].shift(1) < df['ST'].shift(1))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def paraSar(self,df:pd.DataFrame=None, n:int=50, strat_name:str='ParaSAR', 
                exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the crosses upwards the Parabolic SAR indicator while being above 
        the Moving Average.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Average.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''

        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.movingAverage(n=n, method='s', datatype='Close', 
                                           dataname='MA', new_df=df)
        df = self.indicators.sar(af=0.02, amax=0.2, dataname='PSAR', new_df=df)
        
        short_condition = (df['Close'] < df['MA']) & (df['Close'] < df['PSAR']) & \
                        (df['Close'].shift(1) > df['PSAR'].shift(1))
        long_condition = (df['Close'] > df['MA']) & (df['Close'] > df['PSAR']) & \
                        (df['Close'].shift(1) < df['PSAR'].shift(1))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def trendInten(self,df:pd.DataFrame=None, n:int=100, m:int=14, lower:float=40, 
                   upper:float=60, strat_name:str='TrendInten', 
                   exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Trend Intensity crosses upwards the upper level while being above 
        the Moving Average.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Average.
        m: int
            Length of the Trend Intensity oscillator.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.movingAverage(n=n, method='s', datatype='Close', 
                                           dataname='MA', new_df=df)
        df = self.indicators.trendIntensity(n=m, method='e', datatype='Close', 
                                            dataname='TI', new_df=df)
        
        short_condition = (df['Close'] < df['MA']) & (df['TI'] < lower) & \
                        (df['TI'].shift(1) > lower)
        long_condition = (df['Close'] > df['MA']) & (df['TI'] > upper) & \
                        (df['TI'].shift(1) < upper)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def hidDiv(self,df:pd.DataFrame=None, n:int=14, lower:float=20, 
                upper:float=70, width:int=30, strat_name:str='HiddenDiv', 
                exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the Stochastic forms a hidden divergence.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Stochastic Oscillator.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.stochasticOscillator(n=n, m=3, p=3, dataname='SO', 
                                                  new_df=df)

        max_i = len(df)
        long_condition = []
        short_condition = []
        for i,idx  in enumerate(df.index):
            prev_candle = df.iloc[i-1]
            candle = df.iloc[i]
            done = False

            # Long
            if candle['SOK'] < lower and prev_candle['SOK'] > lower:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    high = df.iloc[j]
                    if lower < high['SOK']:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            higher_low = df.iloc[k]
                            if higher_low['SOK'] < lower and \
                                higher_low['Close'] > candle['Close'] and \
                                higher_low['SOK'] < candle['SOK']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if lower < df.iloc[l]['SOK']:
                                        long_condition.append(True)
                                        short_condition.append(False)
                                        done = True
                                        break
                            if done:
                                break
                    if done:
                        break

            # Short
            elif candle['SOK'] > upper and prev_candle['SOK'] < upper:
                for j in range(i+1, i+width if i+width < max_i else max_i):
                    low = df.iloc[j]
                    if low['SOK'] < upper:
                        for k in range(j+1, i+width if i+width < max_i else max_i):
                            lower_high = df.iloc[k]
                            if lower_high['SOK'] > upper and \
                                lower_high['Close'] < candle['Close'] and \
                                lower_high['SOK'] > candle['SOK']:
                                for l in range(k+1, i+width if i+width < max_i else max_i):
                                    if upper > df.iloc[l]['SOK']:
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

        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def rsiNeutrality(self,df:pd.DataFrame=None, n:int=21, m:int=7, lower:float=50, upper:float=50, 
              strat_name:str='RSINeut', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI crosses the RSI by a threshold.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the RSI.
        m: int
            Margin above and below the upper and lower levels for confirmation.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.movingAverage(n=n, method='s', datatype='Close', 
                                        dataname='MA', new_df=df)
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                dataname='RSI', new_df=df)
        
        short_condition = (df['RSI'] < lower - m) & (df['RSI'].shift(1) < lower) & \
                        (df['RSI'].shift(2) > lower) & \
                        (df['RSI'].shift(3) > lower) & \
                        (df['RSI'].shift(4) > lower)
        long_condition = (df['RSI'] > upper + m) & (df['RSI'].shift(1) > upper) & \
                        (df['RSI'].shift(2) < upper) & \
                        (df['RSI'].shift(3) < upper) & \
                        (df['RSI'].shift(4) < upper)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def vamaTrend(self,df:pd.DataFrame=None, n:int=3, m:int=233,
              strat_name:str='VamaTrend', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the price crosses the VAMA plus a threshold based on the ATR.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the VAMA.
        m: int
            Length of the larger volatility for adjusting with the VAMA method.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.atr(n=20, method='s', dataname='ATR', new_df=df)
        df = self.indicators.movingAverage(n=n, m=m, method='vama', datatype='Close', 
                                            dataname='MA', new_df=df)

        max_i = len(df)
        long_condition = []
        short_condition = []
        for i,idx  in enumerate(df.index):
            prev_candle = df.iloc[i-1]
            candle = df.iloc[i]
            done = False

            # Long
            if prev_candle['Close'] < prev_candle['MA'] and candle['Close'] > candle['MA']:
                for j in range(i+1, max_i):
                    next_candle = df.iloc[j]
                    if next_candle['Close'] > next_candle['MA'] + next_candle['ATR']:
                        long_condition.append(True)
                        short_condition.append(False)
                        done = True
                        break
                    elif candle['Close'] < candle['MA']:
                        break

            # Short
            if prev_candle['Close'] > prev_candle['MA'] and candle['Close'] < candle['MA']:
                for j in range(i+1, max_i):
                    next_candle = df.iloc[j]
                    if next_candle['Close'] < next_candle['MA'] - next_candle['ATR']:
                        long_condition.append(True)
                        short_condition.append(False)
                        done = True
                        break
                    elif candle['Close'] > candle['MA']:
                        break
            
            if not done:
                long_condition.append(False)
                short_condition.append(False)

        short_condition = pd.Series(short_condition)
        short_condition.index = df.index
        long_condition = pd.Series(long_condition)
        long_condition.index = df.index
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def superRsiTrend(self,df:pd.DataFrame=None, n:int=14, m:int=20, mult:float=3,
                      lower:float=50, upper:float=50, strat_name:str='SuperRSI', 
                      exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the price crosses upwards the SuperTrend indicator and the RSI is above of 
        its upper level.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the RSI.
        m: int
            Length of the SuperTrend.
        mult: int
            ATR multiplier for the SuperTrend.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df = self.indicators.superTrend(n=m, method='s', mult=mult, datatype='Close', 
                                           dataname='ST', new_df=df)
        df = self.indicators.rsi(n=n, method='e', datatype='Close', 
                                dataname='RSI', new_df=df)
        
        short_condition = (df['Close'] < df['ST']) & (df['Close'].shift(1) > df['ST'].shift(1)) & \
                        (df['RSI'] < lower)
        long_condition = (df['Close'] > df['ST']) & (df['Close'].shift(1) < df['ST'].shift(1)) & \
                        (df['RSI'] > upper)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def catapultTrend(self,df:pd.DataFrame=None, n:int=21, m:int=14, p:int=200,
                      level:float=30, lower:float=50, upper:float=50, 
                      strat_name:str='CatapultTrend', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RVI crosses it's level upwards after being below it for the last 5 periods, 
        the RSI is above it's upper level and the Close is above the Moving Average.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the RVI.
        m: int
            Length of the RSI.
        p: int
            Length of the Moving Average.
        level: float
            RVI limit.
        lower: float
            Lower RSI limit.
        upper: float
            Upper RSI slimit.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

        df['Std'] = df['Close'].rolling(n).std(ddof=0)
        df = self.indicators.rsi(n=n, method='e', datatype='Std', 
                                dataname='RVI', new_df=df)
        df = self.indicators.rsi(n=m, method='e', datatype='Close', 
                                dataname='RSI', new_df=df)
        df = self.indicators.movingAverage(n=p, method='s', datatype='Close', 
                                           dataname='MA', new_df=df)

        vol_condition = (df['RVI'] < level) & (df['RVI'].shift(1) > level) & \
                    (df['RVI'].shift(2) > level) & (df['RVI'].shift(3) > level) & \
                    (df['RVI'].shift(4) > level) & (df['RVI'].shift(5) > level)
        
        short_condition = vol_condition & (df['RSI'] < lower) & (df['Close'] < df['MA'])
        long_condition = vol_condition & (df['RSI'] > upper) & (df['Close'] > df['MA'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
class Signals(PrimaryIndicatorSignals, SecondaryIndicatorSignals, KSignals, ContrarianStrategies, TrendStrategies):
    
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

        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.donchian(n, high_data='High', low_data='Low', dataname='DC', new_df=df)
        
        long_condition = (df['Close'] > df['DCUP'].shift(1))
        short_condition = (df['Close'] < df['DCDN'].shift(1))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        self.ohlc_df = df.copy()

        return self.ohlc_df
    
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        

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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) < df['KCH2']) & \
                                  (df['Close'].shift(2) > df['KCH2'].shift(2)), 1,
                        np.where((df['Close'].shift(1) > df['KCL2']) & \
                                 (df['Close'].shift(2) < df['KCL2'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) < df['KCH2']) & (df['Close'].shift(2) > df['KCH2'].shift(2)), 1,
                        np.where((df['Close'].shift(1) > df['KCL2']) & (df['Close'].shift(2) < df['KCL2'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
        
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
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
        
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
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
        
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
            
        df = self.indicators.atr(n=n, method='s', dataname='ATR', new_df=df)
        df['ATRpct'] = df['ATR'].rolling(n).quantile(quantile)

        short_condition = (df['Close'] < df['Open']) & \
                        (df['ATR'] >= df['ATRpct'])
        long_condition = (df['Close'] > df['Open']) & \
                        (df['ATR'] >= df['ATRpct'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'].shift(1) > df['High'].shift(2)), 1,
                        np.where((df['Close'].shift(1) < df['Low'].shift(2)), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def indexPullback(self,df:pd.DataFrame=None, n:int=200, u:int=7, d:int=5, 
                      strat_name:str='IdxPull', exit_signal:bool=True) -> pd.DataFrame: 
        
        '''
        Buy when si between the moving averages of the Highs and the Lows after being above 
        the Highs moving average the period before and 10 times before.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Averages.
        u: int
            Number of lows above current close.
        d: int
            Number of highs below current close.
        strat_name: str
            Name of the strategy that uses the signal.
        exit_signal: bool
            True to generate an exit signal too.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing all the data.
        '''
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
        df = self.indicators.movingAverage(n=n, method='s', datatype='Close', 
                                            dataname='SMA', new_df=df)
        
        short_condition = (df['High'].rolling(d).max().shift(1) < df['Close']) & \
                        (df['Close'] < df['SMA'])
        long_condition = (df['Low'].rolling(u).min().shift(1) > df['Close']) & \
                        (df['Close'] > df['SMA'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df = self.indicators.movingAverage(n=u, method='s', datatype='Close', 
                                                dataname='UMA', new_df=df)
            df = self.indicators.movingAverage(n=d, method='s', datatype='Close', 
                                                dataname='DMA', new_df=df)
            # Time limit should be 7 days
            long_condition = (df['Close'] > df['UMA'])
            short_condition = (df['Close'] < df['DMA'])
            df[self._renameExit(strat_name)] = np.where(long_condition, 1,
                        np.where(short_condition, -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def goldCorr(self,df:pd.DataFrame=None, n:int=2, lower:float=90, upper:float=10,
                strat_name:str='GoldCorr', exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI is below the 90 level and the gold is making a pullback.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Averages.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols+['GoldClose'], overwrite=True)
        
        df = self.indicators.rsi(n=2, method='s', datatype='Close', 
                                dataname='RSI', new_df=df)
        df = self.indicators.movingAverage(n=10, method='s', datatype='GoldClose', 
                                            dataname='SMA10', new_df=df)
        df = self.indicators.movingAverage(n=50, method='s', datatype='GoldClose', 
                                            dataname='SMA50', new_df=df)
        df = self.indicators.movingAverage(n=200, method='s', datatype='GoldClose', 
                                            dataname='SMA200', new_df=df)
        
        short_condition = (df['RSI'] >= upper) & (df['GoldClose'] >= df['SMA10']) & \
                        (df['GoldClose'] >= df['SMA50']) & (df['SMA50'] < df['SMA200'])
        long_condition = (df['RSI'] <= lower) & (df['GoldClose'] <= df['SMA10']) & \
                        (df['GoldClose'] <= df['SMA50']) & (df['SMA50'] > df['SMA200'])
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df[self._renameEntry(strat_name)].shift(2) > 0), 1,
                        np.where((df[self._renameEntry(strat_name)].shift(2) < 0), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def volatStoch(self,df:pd.DataFrame=None, n:int=14, lower:float=75, upper:float=25,
                    v_level:float=1, strat_name:str='VolS', exit_signal:bool=False
                    ) -> pd.DataFrame: 
        
        '''
        Buy when the RSI is below the 90 level and the gold is making a pullback.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Averages.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols+['Volume'], overwrite=True)
        
        df = self.indicators.stochasticOscillator(n=n, method='s', 
                                                dataname='SO', new_df=df)
        df = self.indicators.bollingerBands(n=20, method='s', desvi=0.5, datatype='Close', 
                                            dataname='BB', new_df=df)
        
        short_condition = (df['BBW'] > v_level) & (df['Volume'] < df['Volume'].shift(1)) & \
                        (df['Open'] >= df['Open'].shift(1)) & (df['SOK'] >= upper)
        long_condition = (df['BBW'] > v_level) & (df['Volume'] < df['Volume'].shift(1)) & \
                        (df['Open'] <= df['Open'].shift(1)) & (df['SOK'] <= lower)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df[self._renameEntry(strat_name)].shift(1) > 0), 1,
                        np.where((df[self._renameEntry(strat_name)].shift(1) < 0), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def dailyPB(self,df:pd.DataFrame=None, strat_name:str='DPB', exit_signal:bool=False
                    ) -> pd.DataFrame: 
        
        '''
        Buy when the RSI is below the 90 level and the gold is making a pullback.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Averages.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols+['DateTime'], overwrite=True)
        
        short_condition = (df['Close'] >= df['Open'].shift(3)) & \
                        (df['Low'].shift(4) >= df['Close'].shift(7)) & \
                        (df['Open'].shift(4) >= df['Open'].shift(8)) & \
                        (df['DateTime'].dt.dayofweek != 3)
        long_condition = (df['Close'] <= df['Open'].shift(3)) & \
                        (df['Low'].shift(4) <= df['Close'].shift(7)) & \
                        (df['Open'].shift(4) <= df['Open'].shift(8)) & \
                        (df['DateTime'].dt.dayofweek != 3)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'] > df['Open']), 1,
                        np.where((df['Close'] < df['Open']), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def volatPB(self,df:pd.DataFrame=None, n:int=14, v_level:float=0, 
                   strat_name:str='VolPB', exit_signal:bool=True) -> pd.DataFrame: 
        
        '''
        Buy when the RSI is below the 90 level and the gold is making a pullback.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Averages.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols+['Volume'], overwrite=True)
        
        df = self.indicators.bollingerBands(n=n, method='s', desvi=0.5, datatype='Close', 
                                            dataname='BB', new_df=df)
        
        short_condition = (df['BBW'] > v_level) & (df['High'] >= df['Open'].shift(1)) & \
                        (df['High'].shift(1) >= df['Low'].shift(5)) & \
                        (df['DateTime'].dt.dayofweek != 1)
        long_condition = (df['BBW'] > v_level) & (df['Low'] <= df['Open'].shift(1)) & \
                        (df['Low'].shift(1) <= df['High'].shift(5)) & \
                        (df['DateTime'].dt.dayofweek != 1)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df[self._renameEntry(strat_name)].shift(1) > 0), 1,
                        np.where((df[self._renameEntry(strat_name)].shift(1) < 0), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def pullbackBounce(self,df:pd.DataFrame=None, strat_name:str='PBB', exit_signal:bool=False
                    ) -> pd.DataFrame: 
        
        '''
        Buy when the RSI is below the 90 level and the gold is making a pullback.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Averages.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols, overwrite=True)
        
        short_condition = (df['Close'].shift(2) >= df['Low'].shift(3)) & \
                        (df['Close'].shift(4) < df['Open'].shift(7)) & \
                        (df['Close'] < df['Close'].shift(1)) & \
                        (df['Close'].shift(1) < df['Close'].shift(2)) & \
                        (df['Open'].shift(4) < df['High'].shift(6))
        long_condition = (df['Close'].shift(2) <= df['High'].shift(3)) & \
                        (df['Close'].shift(4) > df['Open'].shift(7)) & \
                        (df['Close'] > df['Close'].shift(1)) & \
                        (df['Close'].shift(1) > df['Close'].shift(2)) & \
                        (df['Open'].shift(4) > df['Low'].shift(6))
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df['Close'] > df['Open']), 1,
                        np.where((df['Close'] < df['Open']), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    
    def wednesdayStrat(self,df:pd.DataFrame=None, strat_name:str='WedBO', 
                       exit_signal:bool=False) -> pd.DataFrame: 
        
        '''
        Buy when the RSI is below the 90 level and the gold is making a pullback.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the price data.
        n: int
            Length of the Moving Averages.
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
        
        
        df = self._newDf(df, needed_cols=self.needed_cols+['Volume'], overwrite=True)
        
        short_condition = (df['High'] < df['Close'].shift(5)) & \
                        (df['Open'].shift(3) < df['High'].shift(9)) & \
                        (df['Open'].shift(7) < df['Close'].shift(8)) & \
                        (df['DateTime'].dt.dayofweek != 2)
        long_condition = (df['Low'] > df['Close'].shift(5)) & \
                        (df['Open'].shift(3) > df['Low'].shift(9)) & \
                        (df['Open'].shift(7) > df['Close'].shift(8)) & \
                        (df['DateTime'].dt.dayofweek != 2)
        exe_condition = (df['Spread'] < 0.25*df['SLdist']) & \
                        (df['SLdist'] > 0.00001)
        
        df = self._generateSignal(df, strat_name, long_condition, short_condition, 
                                  exe_condition)

        if exit_signal:
            df[self._renameExit(strat_name)] = np.where((df[self._renameEntry(strat_name)].shift(1) > 0), 1,
                        np.where((df[self._renameEntry(strat_name)].shift(1) < 0), -1, 0))

        self.ohlc_df = df

        return self.ohlc_df
    



def plotSignals(df:pd.DataFrame, indicators:list=[]) -> None:

    rows = 1
    row_heights = [5]
    oscillators = []
    for indicator in indicators:
        if 'entry' in indicator.lower() or 'exit' in indicator.lower():
            continue
        if df[indicator].max() < df['Close'].max()/2 or df[indicator].max() > df['Close'].max()*2:
            rows += 1
            row_heights.append(2)
            oscillators.append(indicator)

    if 'volume' in [c.lower() for c in df.columns]:
        fig = make_subplots(rows=rows+1, cols=1, shared_xaxes=True, vertical_spacing=0, row_heights=row_heights+[2])
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
        volume = True
    else:
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0, row_heights=row_heights)

    # Candlesticks
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                        low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)

    # Entry signals
    strat_name = [c for c in df.columns if 'entry' in c.lower()]
    for s in strat_name:
        fig.add_trace(go.Scatter(x=df.index[df[s] > 0], y=df['Open'].shift(-1)[df[s] > 0], name=f'{s}Long', 
                                    marker_color='Blue', marker_symbol='triangle-right', marker_size=15, mode='markers'), 
                        row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[df[s] < 0], y=df['Open'].shift(-1)[df[s] < 0], name=f'{s}Short', 
                                    marker_color='Orange', marker_symbol='triangle-right', marker_size=15, mode='markers'), 
                        row=1, col=1)

    # Indicators
    row = 3 if volume else 2
    for indicator in indicators:
        if 'entry' in indicator.lower() or 'exit' in indicator.lower():
            continue
        if indicator in oscillators:
            fig.add_trace(go.Scatter(x=df.index, y=df[indicator], name=indicator), row=row, col=1)
            row += 1
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df[indicator], name=indicator), row=1, col=1)

    # Formating
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(
        title_text=f'Price',
        autosize=False,
        width=900,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    fig.show()

if __name__ == '__main__':

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    import os

        
    if False:
        from degiro import DeGiro
        degiro = DeGiro('OneMade','Onemade3680')
        products = degiro.getProducts(exchange_id=663,country=846) # Nasdaq exchange
        asset = products.iloc[213] # AAPL -> vwdid = 350015372
        raw = degiro.getPriceData(asset['vwdId'], 'PT1H', 'P5Y', tz='UTC')
    else:
        import yfinance as yf
        raw = yf.Ticker('SPY').history(period='2y',interval='1h')

    raw['SLdist'] = Indicators(raw).atr(n=20, method='s', dataname='ATR')['ATR']

    signals = Signals(raw, backtest=True, errors=False)

    data = signals.catapultTrend(df=raw)
    sign = data[data[data.columns[-1]] != 0]
    prev_indicators = raw.columns
    new_indicators = [c for c in data.columns if c not in prev_indicators]
    plotSignals(data, new_indicators)
