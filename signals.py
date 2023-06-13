
import numpy as np
import pandas as pd

from indicators import Indicators
from config import strategies

class Signals:

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
