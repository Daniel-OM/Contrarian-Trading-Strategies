
import numpy as np
import pandas as pd
import copy

class OHLC:
    
    ''' 
    Class for manipulating OHLC Dataframes.
    '''

    ohlc_df = None

    def __init__(self, ohlc:pd.DataFrame=None, errors:bool=True, 
                 verbose:bool=True) -> None:

        '''
        Function initiate the Indicators class.

        Parameters
        ----------
        ohlc: pd.DataFrame
            DataFrame with OHCL data for an asset. The open columns must 
            be named 'Open', the close column 'Close', the high column
            'High' and the low column 'Low
        
        '''

        self.errors = errors
        self.verbose = verbose
        self._newDf(ohlc, overwrite=True)

    def _deepcopy(self, df:pd.DataFrame) -> pd.DataFrame:

        '''
        Calculates the nth number of the fFibonacci sequence.

        Parameters
        ----------
        n: int
            Number of the sequence to return.

        Returns
        -------
        value: int
            Number of the fibonacci sequence.
        '''
        
        new_df = pd.DataFrame()
        for c in df.columns:
            new_df[c] = df[c].copy(deep=True)
        
        return new_df
    
    def _newDf(self, df:pd.DataFrame, needed_cols=['Open', 'High', 'Low', 'Close'], 
               overwrite:bool=True) -> None:

        '''
        Checks if the new data is in the correct format and contains the needed 
        columns.

        Parameters
        ----------
        df: pd.DataFrame
            Data in Dataframe format.
        needed_cols: list
            List of the columns needed in the Dataframe.
        overwrite: bool
            True to overwrite the current object's data.
        '''

        if isinstance(df, pd.DataFrame):
            rename_dict = {}
            columns = df.columns
            for col in needed_cols: 
                if col.lower() not in [c.lower() for c in columns]:
                    if self.errors:
                        raise ValueError(f'"{col}" is not between the dataframe columns.')
                    else:
                        df[col] = [0]*len(df)
                        if self.verbose:
                            print(f'"{col}" is not between the dataframe columns.')
                elif col not in [c for c in columns]:
                    rename_dict[[c for c in columns if c == col.lower()][0]] = col
            
            if len(rename_dict) > 0:
                df.rename(columns=rename_dict, inplace=True)

            if overwrite:
                self.ohlc_df = df.copy(deep=True)#self._deepcopy(df)
                return self.ohlc_df
            else:
                return df.copy(deep=True)

        elif not isinstance(self.ohlc_df, pd.DataFrame):
            if self.errors:
                raise ValueError('There is no DataFrame with data to use.')
            elif self.verbose:
                print('There is no DataFrame with data to use.')
            return pd.DataFrame()
        else:
            return self.ohlc_df.copy(deep=True)

    def _labelCheck(self, label:str) -> None:

        '''
        Checks if the column exists.

        Parameters
        ----------
        label: str
            Columns name to check.
        '''
        
        if label not in self.ohlc_df.columns:
            raise ValueError('The column name is not in the DataFrame. These ' + \
                             f'are the valid column names: {self.ohlc_df.columns}')


class Charts(OHLC):

    ''' 
    Class for formating the candles to other type of charts.
    '''

    def heiken_ashi(self,openHA:bool=False):

        '''
        Function to obtain the Heiken Ashi candles.

        Parameters
        ----------
        openHA: bool
            If True the candle Open will be based on the previous Heiken Ashi candle, 
            else it will be based on the original previous candle Open.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains the Heiken Ashi formated data. The columns with the new data will 
            have 'HA' before their original name: 'HAOpen', 'HAHigh', 'HALow', 'HAClose'.
        '''

        # If Open based on previous Heiken Ashi Open
        if openHA:
            
            # Create the Close column
            s = len(self.ohlc_df['Close'].tolist())
            HAClose = [np.nan]*s
            for i in range(s):
                if i == 0:
                    HAClose[i] = self.ohlc_df['Close'].tolist()[i]
                else:
                    HAClose[i] = (self.ohlc_df['Open'].tolist()[i] + \
                                  self.ohlc_df['Close'].tolist()[i] + \
                                  self.ohlc_df['High'].tolist()[i] + \
                                  self.ohlc_df['Low'].tolist()[i]) / 4
            self.ohlc_df['HAClose'] = HAClose

            # Create the Open column
            HAOpen = [np.nan]*s
            for i in range(s):
                if i == 0:
                    HAOpen[i] = self.ohlc_df['Open'].tolist()[i]
                else:
                    HAOpen[i] = (HAClose[i-1]+HAOpen[i-1])/2
            self.ohlc_df['HAOpen'] = HAOpen

        # If Open based on the original previous candle
        else:
            self.ohlc_df['HAOpen'] = (self.ohlc_df['Open'].shift(1) + self.ohlc_df['Close'].shift(1)) / 2
            self.ohlc_df['HAClose'] = (self.ohlc_df['Open'] + self.ohlc_df['Close'] + self.ohlc_df['High'] + self.ohlc_df['Low']) / 4
        
        # Create the High and Low columns
        self.ohlc_df['HAHigh'] = np.where(self.ohlc_df['HAClose']>self.ohlc_df['HAOpen'],np.where(self.ohlc_df['HAClose']>self.ohlc_df['High'],self.ohlc_df['HAClose'],self.ohlc_df['High']),
                                np.where(self.ohlc_df['HAOpen']>self.ohlc_df['High'],self.ohlc_df['HAOpen'],self.ohlc_df['High']))
        self.ohlc_df['HALow'] = np.where(self.ohlc_df['HAClose']<self.ohlc_df['HAOpen'],np.where(self.ohlc_df['HAClose']<self.ohlc_df['Low'],self.ohlc_df['HAClose'],self.ohlc_df['Low']),
                                np.where(self.ohlc_df['HAOpen']<self.ohlc_df['Low'],self.ohlc_df['HAOpen'],self.ohlc_df['Low']))
        #self.ohlc_df = self.ohlc_df.drop(['Open','Close','High','Low'],axis=1)

        
        return self.ohlc_df





class Indicators(OHLC):

    ''' 
    Class with all the indicators.
    '''
        
    def fibSequence(self, n:int) -> int:

        '''
        Calculates the nth number of the fFibonacci sequence.

        Parameters
        ----------
        n: int
            Number of the sequence to return.

        Returns
        -------
        value: int
            Number of the fibonacci sequence.
        '''

        if n == 1:
            return 1
        elif n == 0:
            return 0
        else:
            return self.fibSequence(n-1) + self.fibSequence(n-2)
        
    def movingAverage(self, n:int=20, m:int=None, method:str='s', datatype:str='Close', 
            dataname:str=None, new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Moving Average indicator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        m: int
            Length of the larger volatility for adjusting with the VAMA method.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
            - Volatility adjusted: vama
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column containing the indicator values.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the Moving average.
        '''

        df = self._newDf(new_df, overwrite=False)

        # Simple Moving Average
        if method == 's':
            if dataname == None:
                dataname = 'SMA'
            df[dataname] = df[datatype].rolling(n).mean()

        # Exponential Moving Average
        elif method == 'e':
            if dataname == None:
                dataname = 'EMA'
            df[dataname] = df[datatype]\
                            .ewm(span=n,adjust=False,min_periods=n).mean()

        # Weighted Moving Average
        elif method == 'w':
            weights = np.arange(1,n+1)
            if dataname == None:
                dataname = 'LWMA'
            df[dataname] = df[datatype].rolling(n)\
                            .apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
        
        # Volume Weighted Moving Average
        elif method == 'v':
            if 'Volume' in df.columns:
                # df[dataname] = (df[datatype] * df['Volume']) / df['Volume'].rolling(n).cumsum()
                if dataname == None:
                    dataname = 'VWMA'
                df[dataname] = (df[datatype].rolling(n).mean() * df['Volume']) / df['Volume'].rolling(n).cumsum()
            else:
                print('There is no Volume column in the DataFrame passed, a SMA is calculated instead.')
                if dataname == None:
                    dataname = 'SMA'
                df[dataname] = df[datatype].rolling(n).mean()

        # Volume Weighted Average Price
        elif method == 'vwap':
            if dataname == None:
                dataname = 'VWAP'
            df[dataname] = (df['Volume'] * (df['High'] + df['Low']) / 2).cumsum() \
                            / df['Volume'].cumsum()
        
        # Fibonacci Moving Average
        elif method == 'f':
            if dataname == None:
                dataname = 'FMA'
            
            temp_df = pd.DataFrame()
            for i in range(3,n):
                f = self.fibSequence(i)
                temp_df['TempMA'+str(i)] = df[datatype] \
                                        .ewm(span=f,adjust=False,min_periods=f).mean()
                
            temp_df['SUM'] = temp_df.sum(axis=1,skipna=False)
            df[dataname] = temp_df['SUM']/(n-3)

        # Volatility adjusted
        elif method == 'vama':
            if dataname == None:
                dataname = 'VAMA'

            vol1 = df[datatype].rolling(n).std(ddof=0)
            vol2 = df[datatype].rolling(m).std(ddof=0)
            alpha = 0.2 * vol1/vol2
            df[dataname] = alpha * df[datatype] + (1-alpha) * df[datatype].shift(1)

            vama = []
            c = 0
            for i, idx in enumerate(df.index):
                if df[dataname].loc[idx] != df[dataname].loc[idx] or c == 0:
                    vama.append(df[dataname].loc[idx])
                    if vama[-1] == vama[-1]:
                        c += 1
                else:
                    vama.append((alpha.iloc[i] * df[datatype].loc[idx]) + (1-alpha.iloc[i]) * vama[-1])

            df[dataname] = vama

        else:
            print('No method was passed so a SMA is calculated.')
            if dataname == None:
                dataname = 'SMA'
            df[dataname] = df[datatype].rolling(n).mean()

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            return df

    def megaTrend(self, n:int=72, method:str='w', datatype:str='Close', 
                  dataname:str='MegaTrend', new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Mega Trend indicator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s
            - Exponential: e
            - Weighted: w (default)
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column containing the indicator values.
            Default is MegaTrend.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the Dochian Channel bands.
        '''

        df = self._newDf(new_df, overwrite=False)
        
        # Calculate necesary data
        df = self.movingAverage(n=2*n,method=method,datatype=datatype,
                                dataname='MTMA1',df=df)
        df = self.movingAverage(n=n,method=method,datatype=datatype,
                                dataname='MTMA2',df=df)
        # Calculamos the vector
        df['Vect'] = 2*df['MTMA2'] - df['MTMA1']
        # Calculamos the vector average with half the period
        df = self.movingAverage(n=int((2*n)**(1/2)),method=method,datatype='Vect',
                                dataname=dataname,df=df)

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            return df
        
    def vortex(self, n:int=28, dataname:str='Vortex', 
               new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Vortex indicator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        dataname: str
            Name of the resulting columns containing the indicator values.
            Default is Vortex. As two columns will be added the upper band 
            will ne dataname+'UP' and the lower band dataname+'DN'.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the Vortex bands.
        '''

        df = self._newDf(new_df, overwrite=False)

        # Calculamos datos necesarios
        df = self.atr(n=1,dataname='TempATR', new_df=df)

        df['SumUpVM'] = abs(df['High']-df['Low'].shift(1)).roling(n).sum()
        df['SumDnVM'] = abs(df['Low']-df['High'].shift(1)).roling(n).sum()
        df['SumATRVM'] = df['TempATR'].roling(n).sum()

        df[dataname+'UP'] = df['SumUpVM'] / df['SumATRVM']
        df[dataname+'DN'] = df['SumDnVM'] / df['SumATRVM']            
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname+'UP'] = df[dataname+'UP']
            self.ohlc_df[dataname+'DN'] = df[dataname+'DN']
            return self.ohlc_df
        
        else:
            return df

    def envelopes(self, n:int=10, method:str='s', k:float=0.1,
                  datatype:str='Close', dataname:str='Env', 
                  new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Envelopes indicator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        k: float
            Percentage of price to get as bands amplitude. This number 
            will be divided by 100 to get the per unit multiplier.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns containing the indicator values.
            Default is Env. As two columns will be added the upper band 
            will ne dataname+'UP' and the lower band dataname+'DN'.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the Envelopes bands.
        '''

        df = self._newDf(new_df, overwrite=False)

        df = self.movingAverage(n=n,method=method,datatype=datatype,
                                dataname=dataname+'MA', new_df=df)

        df[dataname+'UP'] = df[dataname+'MA']*(1+k/100)
        df[dataname+'DN'] = df[dataname+'MA']*(1-k/100)
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname+'UP'] = df[dataname+'UP']
            self.ohlc_df[dataname+'DN'] = df[dataname+'DN']
            return self.ohlc_df
        
        else:
            return df
     
    def forexTradeOscillator(self, n:int=9, f1:float=2/3, f2:float=1/3,
                             rsv:bool=False, dataname:str='ForexTrade',
                             new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Envelopes indicator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        k: int
            Percentage of price to get as bands amplitude. This number 
            will be divided by 100 to get the per unit multiplier.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns containing the indicator values.
            Default is ForexTrade. As there are three lines names will be: 
            dataname+'K%', dataname+'D%' and dataname+'J%'.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the Forex Trades three 
            lines.
        '''

        df = self._newDf(new_df, overwrite=False)
        
        close = df['Close'].tolist()
        low = df['Low'].tolist()
        high = df['High'].tolist()
        Ln = [np.nan]*len(close)
        Hn = [np.nan]*len(close)
        for i in range(n,len(close)):
            Ln[i] = close[i]
            Hn[i] = close[i]
            for k in range(n):
                if Ln[i] > low[i-k]:
                    Ln[i] = low[i-k]
                if high[i-k] > Hn[i]:
                    Hn[i] = high[i-k]
        df['Ln'] = Ln
        df['Hn'] = Hn
        
        df[dataname+'RSV'] = np.where(df['Hn']-df['Ln'] == 0,50,(df['Close']-df['Ln'])/(df['Hn']-df['Ln'])*100)

        rsv = df[dataname+'RSV'].tolist()
        k = [np.nan]*len(close)
        d = [np.nan]*len(close)
        for i in range(len(close)):
            if i<=n+1:
                k[i] = f1*50+f2*rsv[i]
                d[i] = f1*50+f2*k[i]
            else:
                k[i] = f1*k[i-1]+f2*rsv[i]
                d[i] = f1*d[i-1]+f2*k[i]

        df[dataname+'K%'] = k
        df[dataname+'D%'] = d
        df[dataname+'J%'] = 3*df[dataname+'D%']-2*df[dataname+'K%'] 

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname+'K%'] = df[dataname+'K%']
            self.ohlc_df[dataname+'D%'] = df[dataname+'D%']
            self.ohlc_df[dataname+'J%'] = df[dataname+'J%']
            if rsv:
                self.ohlc_df[dataname+'RSV'] = df[dataname+'RSV']

            return self.ohlc_df
        
        else:
            if rsv:
                df.drop(['Ln','Hn'],axis=1,inplace=True)
            else:
                df.drop(['Ln','Hn',dataname+'RSV'],axis=1,inplace=True)

            return df
    
    def rateOfChange(self, n:int=1, datatype:str='Close', 
                    dataname:str='ROC', pct:bool=True,
                    new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Rate of Change indicator.

        Parameters
        ----------
        n: int
            Window to calculate the Rate of Change.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column containing the indicator values.
            Default is ROC.
        pct: bool
            True to return data as percentage, multiplied by 100.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the Rate of Change.
        '''

        df = self._newDf(new_df, overwrite=False)

        df[dataname] = (df[datatype]/df[datatype].shift(n) - 1)
        if pct:
            df[dataname] = df[dataname] * 100

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            return df

    def efficiencyRatio(self, n:int=10, datatype:str='Close', 
                        dataname:str='ER', new_df:pd.DataFrame=None
                        ) -> pd.DataFrame:

        '''
        Calculates the Efficiency Ratio indicator.

        Parameters
        ----------
        n: int
            Window to calculate the Efficiency Ratio.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column containing the indicator values.
            Default is ER.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the Efficiency Ratio.
        '''

        df = self._newDf(new_df, overwrite=False)

        df['change'] = abs(df[datatype]-df[datatype].shift(n))
        df['volatility'] = abs(df[datatype]-df[datatype].shift(1))
        df['volSum'] = df['volatility'].rolling(n).sum()
        df[dataname] = df['change']/df['volSum']

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            return df

    def kama(self, n:int=10, scf:int=2, scs:int=30, datatype:str='Close', 
             dataname:str='KAMA', new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Kaufman Adaptative Moving Average indicator.

        Parameters
        ----------
        n: int
            Window to calculate the KAMA.
        scf: int
            Fast smooth constant for the smoothing constant calculation.
        scs: int
            Slow smooth constant for the smoothing constant calculation.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column containing the indicator values.
            Default is KAMA.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the KAMA.
        '''

        df = self._newDf(new_df, overwrite=False)
    
        # Calculamos los datos principales
        df = self.efficiencyRatio(n=n, datatype=datatype, dataname='ER', 
                                  new_df=df)
        df['SC'] = (df['ER']*(2.0/(scf+1.0)-2.0/(scs+1.0))+2.0/(scs+1.0))**2.0

        # Calculamos la media KAMA, como depende de sus valores iniciales se hace así
        sc = df['SC'].tolist()
        data = df[datatype].tolist()
        kama = [0]*len(sc)
        first_value = True
        for i in range(len(sc)):
            if sc[i] != sc[i]:
                kama[i] = np.nan
            else:
                if first_value:
                    kama[i] = data[i]
                    first_value = False
                else:
                    kama[i] = kama[i-1] + sc[i] * (data[i] - kama[i-1])
        
        # Introducimos la lista en el DataFrame
        df[dataname] = kama

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            return df

    def slope(self, n:int=1, datatype:str='Close', dataname:str='Slope', 
              pct:bool=False, new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Rate of Slope indicator.

        Parameters
        ----------
        n: int
            Window to calculate the Slope.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column containing the indicator values.
            Default is Slope.
        pct: bool
            True to return the value as percentage instead of per unit.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the Slope.
        '''

        df = self._newDf(new_df, overwrite=False)

        df[dataname] = (df[datatype] - df[datatype].shift(n)) / n

        if pct:
            df[dataname] = 100*df[dataname]

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            return df
        
    def atr(self, n:int=20, method:str='s', dataname:str='ATR', 
            tr:bool=False, new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Average True Range.

        Parameters
        ----------
        n: int
            Length of the moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        dataname: str
            Name of the resulting column in the DataFrame with the ATR data.
        tr: bool
            True to return the TR column in the DataFrame. The column 
            name would be dataname+'TR'.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the ATR.
        '''

        df = self._newDf(new_df, overwrite=False)

        # Definition of data needed for the TR
        df['H-L'] = abs(df['High']-df['Low'])
        df['H-PC'] = abs(df['High']-df['Close'].shift(1))
        df['L-PC'] = abs(df['Low']-df['Close'].shift(1))

        # Calculation of TR and Moving Average
        df[dataname+'TR'] = df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
        df = self.movingAverage(n=n, method=method, datatype=dataname+'TR', 
                                dataname=dataname, new_df=df)

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            if tr:
                self.ohlc_df[dataname+'TR'] = df[dataname+'TR']
            return self.ohlc_df
        
        else:
            if tr:
                df = df.drop(['H-L','H-PC','L-PC'],axis=1)
            else:
                df = df.drop(['H-L','H-PC','L-PC',dataname+'TR'],axis=1)
            return df
    
    def acr(self, n:int=20, method:str='s', dataname:str='ACR', 
            pct:bool=False, new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Average Candle Range.

        Parameters
        ----------
        n: int
            Length of the moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is ACR.
        pct: bool
            True to return the value as percentage instead of per unit.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the ACR.
        '''

        df = self._newDf(new_df, overwrite=False)
            
        df['ratio'] = df['High']/df['Low']
        df = self.movingAverage(n=n,method=method,datatype='ratio',
                                dataname='TempMA', new_df=df)
        df = df.drop(['ratio'],axis=1)
        df[dataname] = (df['TempMA']-1)

        if pct:
            df[dataname] = df[dataname] * 100
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            return df

    def superTrend(self, n:int=20, method:str='s', mult:float=2.0, 
                   datatype:str='Close', dataname:str='SuperTrend', 
                   new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Super Trend indicator.

        Parameters
        ----------
        n: int
            Length of the moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        mult: float
            Multiplier for the basic bands amplitude.
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is Super Trend.
        pct: bool
            True to return the value as percentage instead of per unit.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the Super Trend.
        '''

        df = self._newDf(new_df, overwrite=False)

        # Get the ATR
        if 'ATR' not in df.columns:
            df = self.atr(n=n,method=method,dataname='ATR', new_df=df)

        # Calculate the basic bands
        df['basic_up'] = (df['High']+df['Low'])/2+mult*df['ATR']
        df['basic_down'] = (df['High']+df['Low'])/2-mult*df['ATR']

        # Calculate the indicator
        data = df[datatype].tolist()

        # Calculate upper band
        bu = df['basic_up'].tolist()
        fu = [0]*len(bu)
        first_up_value = True
        for i in range(len(bu)):
            if bu[i] != bu[i]:
                fu[i] = np.nan
            else:
                if first_up_value:
                    fu[i] = data[i]
                    first_up_value = False
                else:
                    if bu[i]<fu[i-1] and data[i-1]>fu[i-1]:
                        fu[i] = bu[i]
                    else:
                        fu[i] = fu[i-1]

        # Calculate lower band
        bd = df['basic_down'].tolist()
        fd = [0]*len(bd)
        first_down_value = True
        for i in range(len(bd)):
            if bd[i] != bd[i]:
                fd[i] = np.nan
            else:
                if first_down_value:
                    fd[i] = data[i]
                    first_down_value = False
                else:
                    if bd[i]>fd[i-1] and data[i-1]<fd[i-1]:
                        fd[i] = bd[i]
                    else:
                        fd[i] = fd[i-1]
        
        # Get super trend
        supertrend = [0]*len(data)
        for i in range(len(supertrend)):
            if data[i] <= fu[i]:
                supertrend[i] = fu[i]
            else:
                supertrend[i] = fd[i]

        # Store the data
        df[dataname] = supertrend
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['basic_up','basic_down'],axis=1)
            return df
        
    def donchian(self, n:int=70, high_data:str='High', low_data:str='Low',
                dataname:str='DC', new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Donchian Channel.

        Parameters
        ----------
        n: int
            Length of the moving average.
        high_data: str
            DataFrame column used to calculate the channel upper band.
            Default is High.
        low_data: str
            DataFrame column used to calculate the channel lower band.
            Default is Low.
        dataname: str
            Name of the resulting columns in the DataFrame with the data.
            Default is DC. The suffix 'UP' and 'DN' will be added for the 
            upper and lower bands.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the DC.
        '''

        df = self._newDf(new_df, overwrite=False)

        df[dataname+'UP'] = df[high_data].rolling(n).max() 
        df[dataname+'DN'] = df[low_data].rolling(n).min() 
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname+'UP'] = df[dataname+'UP']
            self.ohlc_df[dataname+'DN'] = df[dataname+'DN']
            return self.ohlc_df
        
        else:
            return df

    def vamaBands(self, n:int=20, m:int=100, p:int=30, desvi:float=2., 
                    datatype:str='Close', dataname:str='VB',
                    new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Bollinger Bands.

        Parameters
        ----------
        n: int
            Length of the moving average.
        m: int
            Length of the larger volatility for adjusting with the VAMA method.
        p: int
            Length for the volatility.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
            - Volatility adjusted: vama
        desvi: float
            Multiplier for the bands amplitude.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns in the DataFrame with the data.
            Default is BB. The suffix 'UP', 'DN' and 'W' will be added for the 
            upper and lower band and the width of the bands.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the upper, lower and width 
            of the bands.
        '''

        df = self._newDf(new_df, overwrite=False)

        df = self.movingAverage(n=n, m=m, method='vama', datatype=datatype, 
                                dataname='TempMA', new_df=df)
        std = df[datatype].rolling(p).std(ddof=0)
        df[dataname+'UP'] = df['TempMA'] + desvi*std
        df[dataname+'DN'] = df['TempMA'] - desvi*std
        df[dataname+'W'] = df[dataname+'UP'] - df[dataname+'DN']
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname+'UP'] = df[dataname+'UP']
            self.ohlc_df[dataname+'DN'] = df[dataname+'DN']
            self.ohlc_df[dataname+'W'] = df[dataname+'W']
            return self.ohlc_df
        
        else:
            df = df.drop(['TempMA'],axis=1)
            return df

    def bollingerBands(self, n:int=20, method:str='s', desvi:float=2., 
                       datatype:str='Close', dataname:str='BB',
                       new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Bollinger Bands.

        Parameters
        ----------
        n: int
            Length of the moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        desvi: float
            Multiplier for the bands amplitude.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns in the DataFrame with the data.
            Default is BB. The suffix 'UP', 'DN' and 'W' will be added for the 
            upper and lower band and the width of the bands.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the upper, lower and width 
            of the bands.
        '''

        df = self._newDf(new_df, overwrite=False)

        df = self.movingAverage(n=n, method=method, datatype=datatype, 
                                dataname='TempMA', new_df=df)
        df[dataname+'UP'] = df['TempMA'] + desvi*df[datatype].rolling(n).std(ddof=0) #ddof=0 es necesario porque necesitamos la desviación estandar de la población y no la muestra
        df[dataname+'DN'] = df['TempMA'] - desvi*df[datatype].rolling(n).std(ddof=0) #ddof=0 es necesario porque necesitamos la desviación estandar de la población y no la muestra
        df[dataname+'W'] = df[dataname+'UP'] - df[dataname+'DN']
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname+'UP'] = df[dataname+'UP']
            self.ohlc_df[dataname+'DN'] = df[dataname+'DN']
            self.ohlc_df[dataname+'W'] = df[dataname+'W']
            return self.ohlc_df
        
        else:
            df = df.drop(['TempMA'],axis=1)
            return df

    def pctBollinger(self, n:int=20, desvi:float=2., datatype='Close', 
                      dataname:str='%BB', new_df:pd.DataFrame=None
                      ) -> pd.DataFrame:

        '''
        Calculates the percentage of the Bollinger Bands where the price 
        is from the upper band.

        Parameters
        ----------
        n: int
            Length of the moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        desvi: float
            Multiplier for the bands amplitude.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns in the DataFrame with the data.
            Default is %BB.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the %BB.
        '''

        df = self._newDf(new_df, overwrite=False)

        # Calculamos las bandas de bollinger
        df = self.bollingerBands(n=n, desvi=desvi, datatype=datatype, 
                                      dataname='BB', new_df=df)
        # Calculamos el porcentaje en el que se encuentra el precio desde la banda superior respecto el ancho de la banda
        df[dataname] = df[datatype] - df['BBUP']/df['BBW']*100
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['BBUP', 'BBDN', 'BBW'],axis=1)
            return df

    def macd(self, a:int=12, b:int=26, c:int=9, method:str='e', 
             datatype='Close', dataname:str='MACD', 
             new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Moving Average Convergence Divergence indicator.

        Parameters
        ----------
        a: int
            Length of the fast moving average.
        b: int
            Length of the slow moving average.
        c: int
            Length of the difference moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns in the DataFrame with the data.
            Default is MACD. The suffix 'S' will be added for the 
            signal line.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the MACD and the MACD 
            signal.
        '''

        df = self._newDf(new_df, overwrite=False)
            
        # Calculamos medias móviles exponenciales
        df = self.movingAverage(n=a, method=method, datatype=datatype, dataname='MA_Fast', new_df = df)
        df = self.movingAverage(n=b, method=method, datatype=datatype, dataname='MA_Slow', new_df = df)
        # Hallamos la diferencia entre las dos medias móviles
        df[dataname] = df['MA_Fast']-df['MA_Slow']
        # Hacemos la media móvil exponencial de la diferencia 
        df = self.movingAverage(n=c, method=method, datatype=dataname, dataname=dataname+'S', new_df = df)

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            self.ohlc_df[dataname+'S'] = df[dataname+'S']
            return self.ohlc_df
        
        else:
            df = df.drop(['MA_Fast','MA_Slow'],axis=1)
            return df

    def rsi(self, n:int=14, method:str='e', 
             datatype:str='Close', dataname:str='RSI', 
             new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Relative Strength Index indicator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns in the DataFrame with the data.
            Default is RSI.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the RSI.
        '''

        df = self._newDf(new_df, overwrite=False)
            
        df['delta'] = df[datatype] - df[datatype].shift(1)
        df['gain'] = np.where(df['delta']>=0, df['delta'], 0)
        df['loss'] = np.where(df['delta']<0, abs(df['delta']), 0)

        # Construimos vectores con los valores de pérdida y ganancia media
        df = self.movingAverage(n=n, method=method, datatype='gain', 
                                            dataname='avg_gain', new_df=df)
        df = self.movingAverage(n=n, method=method, datatype='loss', 
                                            dataname='avg_loss', new_df=df)
        # Calculamos el RS (Relative Strength)
        df['RS'] = df['avg_gain']/df['avg_loss']
        # Lo referimos a un rango máximo de 100
        df[dataname] = 100 - (100 / (1 + df['RS']))

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['delta','gain','loss','avg_gain','avg_loss','RS'], axis=1)
            return df

    def rsiAtr(self, n:int=14, m:int=14, o:int=14, method:str='e', 
             datatype='Close', dataname:str='RSIATR', 
             new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the RSI of the RSI/ATR.

        Parameters
        ----------
        n: int
            Length of the ATR.
        m: int
            Length of the first RSI.
        o: int
            Length of the second RSI.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns in the DataFrame with the data.
            Default is RSI.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the RSI.
        '''

        df = self._newDf(new_df, overwrite=False)

        df = self.atr(n=n, method=method, dataname=dataname+'ATR', new_df=df)
        df = self.rsi(n=m, method=method, datatype=datatype, dataname=dataname+'RSI', 
                      new_df=df)
        df[dataname] = df[dataname+'RSI'] / df[dataname+'ATR']

        df = self.rsi(n=o, method=method, datatype=dataname, dataname=dataname, new_df=df)

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop([dataname+'RSI',dataname+'ATR'], axis=1)
            return df

    def adx(self, n:int=20, method:str='e', dataname:str='ADX', 
            new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Average Directional Index indicator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        method: str
            Calculation method used for the moving averages. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        dataname: str
            Name of the resulting columns in the DataFrame with the data.
            Default is ADX.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the ADX.
        '''

        df = self._newDf(new_df, overwrite=False)

        df = self.atr(n=n, method=method, dataname='ATR', tr=False, new_df=df)

        # Si la diferencia entre el máximo actual y el anterior es superior a 
        # la del mínimo actual y el anterior ponemos la diferencia de máximos
        df[dataname+'DMplus'] = np.where((df['High']-df['High'].shift(1)) > \
                                         (df['Low'].shift(1)-df['Low']), 
                                         df['High']-df['High'].shift(1), 0)
        df[dataname+'DMplus'] = np.where(df[dataname+'DMplus']<0, 0, 
                                         df[dataname+'DMplus'])
        
        # Si la diferencia entre el máximo actual y el anterior es inferior a 
        # la del mínimo actual y el anterior ponemos la diferencia de mínimos
        df[dataname+'DMminus'] = np.where((df['Low'].shift(1)-df['Low']) > \
                                          (df['High']-df['High'].shift(1)), 
                                          df['Low'].shift(1)-df['Low'], 0)
        df[dataname+'DMminus'] = np.where(df[dataname+'DMminus']<0, 0, 
                                          df[dataname+'DMminus'])
        
        # Definimos vectores con los valores calculados hasta ahora
        df = self.movingAverage(n=n, method=method, datatype=dataname+'DMplus', 
                                dataname=dataname+'DMplusN', new_df=df)
        df = self.movingAverage(n=n, method=method, datatype=dataname+'DMminus', 
                                dataname=dataname+'DMminusN', new_df=df)
        
        # Escalamos los valores para un rango de 100
        df[dataname+'DIplus'] = 100*(df[dataname+'DMplusN']/df['ATR'])
        df[dataname+'DIminus'] = 100*(df[dataname+'DMminusN']/df['ATR'])
        # Calculamos la diferencia (en valor absoluto) y la suma
        df[dataname+'DIdiff'] = df[dataname+'DIplus'] - df[dataname+'DIminus']
        df[dataname+'DIsum'] = df[dataname+'DIplus'] + df[dataname+'DIminus']
        # Volvemos a escalar
        df[dataname+'DX'] = 100*(abs(df[dataname+'DIdiff'])/df[dataname+'DIsum'])
        df = self.movingAverage(n=n, method=method, datatype=dataname+'DX', 
                                dataname=dataname, new_df=df)

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname+'DIminus'] = df[dataname+'DIminus']
            self.ohlc_df[dataname+'DIplus'] = df[dataname+'DIplus']
            self.ohlc_df[dataname+'DIdiff'] = df[dataname+'DIdiff']
            self.ohlc_df[dataname+'DX'] = df[dataname+'DX']
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop([dataname+'DMplus',dataname+'DMminus','ATR',
                         dataname+'DMplusN',dataname+'DMminusN', dataname+'DIsum'],
                         axis=1)
            return df

    def stochasticOscillator(self, n:int=5, m:int=3, p:int=3, method:str='s', 
                             dataname:str='SO', datatype:str=None, new_df:pd.DataFrame=None
                             ) -> pd.DataFrame:

        '''
        Calculates the Stochastic Oscillator indicator.

        Parameters
        ----------
        n: int
            Length of the maximum and minimum rolling window.
        m: int
            Length of the slow moving average.
        p: int
            Length of the fast moving average.
        method: str
            Calculation method used for the moving averages. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns in the DataFrame with the data.
            Default is SO. A sufix will be added as there are two lines: 
            dataname + 'K' and dataname + 'D'.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the SO.
        '''

        df = self._newDf(new_df, overwrite=False)

        if datatype == None:
            df['k'] = ((df['Close'] - df['Low'].rolling(n).min()) / \
                        (df['High'].rolling(n).max()-df['Low'].rolling(n).min()))*100
        else:
            df['k'] = ((df[datatype] - df[datatype].rolling(n).min()) / \
                        (df[datatype].rolling(n).max()-df[datatype].rolling(n).min()))*100
            
        df = self.movingAverage(n=m, method=method, datatype='k', 
                                dataname=dataname+'K', new_df=df)
        df = self.movingAverage(n=p, method=method, datatype='k', 
                                dataname=dataname+'D', new_df=df)
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname+'K'] = df[dataname+'K']
            self.ohlc_df[dataname+'D'] = df[dataname+'D']
            return self.ohlc_df
        
        else:
            df = df.drop(['k'],axis=1)
            return df
    
    def momentumOscillator(self, n:int=5, datatype:str='Close', 
                            dataname:str='MO', new_df:pd.DataFrame=None
                            ) -> pd.DataFrame:

        '''
        Calculates the Momentum Oscillator indicator.

        Parameters
        ----------
        n: int
            Length of the maximum and minimum rolling window.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns in the DataFrame with the data.
            Default is MO.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the MO.
        '''

        df = self._newDf(new_df, overwrite=False)

        df[dataname] = (df[datatype] / df[datatype].shift(n))*100
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            return df
    
    def awesomeOscillator(self, n:int=5, m:int=34, method:str='s', 
                          dataname:str='AO', new_df:pd.DataFrame=None
                         ) -> pd.DataFrame:

        '''
        Calculates the Awesome Oscillator indicator.

        Parameters
        ----------
        n: int
            Length of the fast moving average.
        m: int
            Length of the slow moving average.
        method: str
            Calculation method used for the moving averages. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is AO.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the AO.
        '''

        df = self._newDf(new_df, overwrite=False)

        df['Midpoint'] = (df['High']+df['Low'])/2
        df = self.movingAverage(n=n, method=method, datatype='Midpoint', 
                                dataname='MA_fast', new_df=df)
        df = self.movingAverage(n=m, method=method, datatype='Midpoint', 
                                dataname='MA_slow', new_df=df)
        df[dataname] = df['MA_fast'] - df['MA_slow']
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['Midpoint', 'MA_fast', 'MA_slow'],axis=1)
            return df

    def lameOscillator(self, n:int=5, m:int=34, method:str='s', 
                       dataname:str='LO', new_df:pd.DataFrame=None
                      ) -> pd.DataFrame:

        '''
        Calculates the Lame Oscillator indicator, which is the inverse 
        of the Awesome Oscillator.

        Parameters
        ----------
        n: int
            Length of the fast moving average.
        m: int
            Length of the slow moving average.
        method: str
            Calculation method used for the moving averages. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is LO.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the LO.
        '''

        df = self._newDf(new_df, overwrite=False)

        df['Midpoint'] = (df['High']+df['Low'])/2
        df = self.movingAverage(n=n, method=method, datatype='Midpoint', 
                                dataname='MA_fast', new_df=df)
        df = self.movingAverage(n=m, method=method, datatype='Midpoint', 
                                dataname='MA_slow', new_df=df)
        df[dataname] = df['MA_slow'] - df['MA_fast']
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['Midpoint', 'MA_fast', 'MA_slow'],axis=1)
            return df

    def accelerationOscillator(self, n:int=5, m:int=34, method:str='s', 
                       dataname:str='AccO', new_df:pd.DataFrame=None
                      ) -> pd.DataFrame:

        '''
        Calculates the Acceleration Oscillator indicator, which is the 
        difference between the Awesome Oscillator and its moving average.

        Parameters
        ----------
        n: int
            Length of the fast moving average.
        m: int
            Length of the slow moving average.
        method: str
            Calculation method used for the moving averages. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is AccO.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the AccO.
        '''

        df = self._newDf(new_df, overwrite=False)


        df = self.awesomeOscillator(n=n, m=m, method=method, dataname='AO', 
                                    new_df=df)
        df = self.movingAverage(n=n, method=method, datatype='AO', 
                                dataname='AOM', new_df=df)
        df[dataname] = df['AO'] - df['AOM']
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['AO', 'AOM'],axis=1)
            return df

    def cci(self, n:int=14, method:str='s', dataname:str='CCI', 
            new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Commodity Channel Index indicator.

        Parameters
        ----------
        n: int
            Length of the moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is CCI.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the CCI.
        '''

        df = self._newDf(new_df, overwrite=False)

        df['TipicalPrice'] = (df['Close'] + df['High'] + df['Low']) / 3
        df = self.movingAverage(n=n, method=method, datatype='TipicalPrice', 
                                dataname='SMATipicalPrice', new_df=df)
        df['Deviation'] = df['TipicalPrice'] \
                            .rolling(min_periods=1, center=False, window=n) \
                            .apply(lambda x: np.fabs(x - x.mean()).mean())

        df[dataname] = (df['TipicalPrice'] - df['SMATipicalPrice']) \
                        / (.015 * df['Deviation'])
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['TipicalPrice', 'SMATipicalPrice', 'Deviation'],
                        axis=1)
            return df

    def williamOscillator(self, n:int=14, dataname:str='W%', 
                        new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the William% Index indicator.

        Parameters
        ----------
        n: int
            Length of the moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is W%.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the W%.
        '''

        df = self._newDf(new_df, overwrite=False)

        # Calculamos los datos necesarios
        df['Highest'] = df['High'].rolling(n).max()
        df['Lowest'] = df['Low'].rolling(n).min()

        # Calculamos el indicador
        # df[dataname] = (1 - (df['Highest'] - df['Close']) / \
        #                 (df['Highest']-df['Lowest'])) * -100 # Between 0 and -100
        df[dataname] = (1 - (df['Highest'] - df['Close']) / \
                        (df['Highest']-df['Lowest'])) * 100 # Between 0 and 100
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['Highest','Lowest'],axis=1)
            return df

    def keltnerChannel(self, n:int=40, mamethod:str='s', atrn:int=40, 
                       atrmethod:str='s', multiplier:float=2,
                       datatype:str='Close', dataname:str='KC', 
                       new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Keltner Channel indicator.

        Parameters
        ----------
        n: int
            Length of the moving average.
        mamethod: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        atrn: int
            Length of the ATR.
        atrmethod: str
            Calculation method used for the ATR. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        multiplier: float
            Number of time to use the ATR for the bands amplitude.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is KC. There will be three final columns named:
            dataname + 'UP', dataname + 'DN' and dataname + 'M'.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the KC.
        '''

        df = self._newDf(new_df, overwrite=False)

        # Calculamos el ATR
        df = self.atr(n=atrn, method=atrmethod, dataname='KATR', new_df=df)

        # Calculamos la media movil
        df = self.movingAverage(n=n, method=mamethod, datatype=datatype, 
                                dataname=dataname+'M', new_df=df)

        # Calculamos el canal de keltner
        df[dataname+'UP'] = df[dataname+'M'] + multiplier*df['KATR']
        df[dataname+'DN'] = df[dataname+'M'] - multiplier*df['KATR']
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname+'DN'] = df[dataname+'DN']
            self.ohlc_df[dataname+'M'] = df[dataname+'M']
            self.ohlc_df[dataname+'UP'] = df[dataname+'UP']
            return self.ohlc_df
        
        else:
            df = df.drop(['KATR'],axis=1)
            return df

    def volatilityBands(self, n:int=20, multiplier:float=2,
                       datatype:str='Close', dataname:str='VB', 
                       new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Volatility Bands indicator.

        Parameters
        ----------
        n: int
            Length of the bands.
        multiplier: float
            Number of time to use the volatility for the bands amplitude.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is VB. There will be three final columns named:
            dataname + 'UP' and dataname + 'DN'.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the VB.
        '''

        df = self._newDf(new_df, overwrite=False)

        # Calculamos la mediana
        df['Median'] = (df['High'].rolling(n).max() + df['Low'].rolling(n).min())/2
        df['Std'] = df[datatype].rolling(n).std(ddof=0)
        df['MaxStd'] = df['Std'].rolling(n).max()
        df[dataname+'DN'] = df['Median'] - multiplier * df['MaxStd']
        df[dataname+'UP'] = df['Median'] + multiplier * df['MaxStd']
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname+'DN'] = df[dataname+'DN']
            self.ohlc_df[dataname+'UP'] = df[dataname+'UP']
            return self.ohlc_df
        
        else:
            df = df.drop(['Median','Std','MaxStd'],axis=1)
            return df

    def simpleFisher(self, n:int=10, m:int=3, p:int=3, method:str='s', 
                     dataname='SFisher', new_df:pd.DataFrame=None
                     ) -> pd.DataFrame:

        '''
        Calculates the Simple Fisher Transform indicator.

        Parameters
        ----------
        n: int
            Length of the dochian channel.
        m: int
            Length of the slow moving average.
        p: int
            Length of the fast moving average.
        method: str
            Calculation method used for the moving averages. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is FT.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the FT.
        '''

        df = self._newDf(new_df, overwrite=False)

        df = self.stochasticOscillator(n=n, m=m, p=p, method=method, dataname='SFSO', 
                                       new_df=df)

        df['SO'+dataname] = df['SFSOK']/100
        df['SO'+dataname] = 2*df['SO'+dataname] - 1

        df['SO'+dataname] = np.where(df['SO'+dataname] == 1, 0.999, 
                            np.where(df['SO'+dataname] == -1, -0.999, 
                                     df['SO'+dataname]))
        
        df[dataname] = 0.5 * np.log((1 + df['SO'+dataname])/(1 - df['SO'+dataname]))
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['SFSOK','SFSOD','SO'+dataname],
                          axis=1)
            return df
        
    def fisher(self,n:int=10,alpha:float=0.33,dataname='Fisher', 
                new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Fisher Transform indicator.

        Parameters
        ----------
        n: int
            Length of the dochian channel.
        alpha: float
            Multiplier for the value calculation.
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is FT.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the FT.
        '''

        df = self._newDf(new_df, overwrite=False)

        df['Median'+dataname] = (df['High'] + df['Low']) / 2
        df['max'+dataname] = df['Median'+dataname].rolling(n).max()
        df['min'+dataname] = df['Median'+dataname].rolling(n).min()
        df['Value1'+dataname] = pd.Series([np.nan]*len(df))
        df[dataname] = pd.Series([np.nan]*len(df))
        df['Value1'+dataname] = alpha * 2.0 * \
            ((df['Close'] - df['min'+dataname]) / \
            np.where(df['max'+dataname] - df['min'+dataname] > 0.5,
                    df['max'+dataname] - df['min'+dataname], 0.501) \
            - 0.5) + (1 - alpha) * \
            np.where(df['Value1'+dataname].isnull(), 0,
                    df['Value1'+dataname].shift(1))
        
        df['Value1'+dataname] = np.where(df['Value1'+dataname]>0.9999,
                                        0.9999,
                                np.where(df['Value1'+dataname]<-0.9999,
                                        -0.9999, df['Value1'+dataname]))
        
        df[dataname] = 0.5 * np.log( (1+df['Value1'+dataname]) / \
                            (1-df['Value1'+dataname]) ) + 0.5* \
                        np.where(df[dataname].isnull(),
                                    0, df[dataname].shift(1))
        df[dataname+'Signal'] = df[dataname].shift(1)
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            self.ohlc_df[dataname+'Signal'] = df[dataname+'Signal']
            return self.ohlc_df
        
        else:
            df = df.drop(['Median'+dataname,'max'+dataname,
                          'min'+dataname,'Value1'+dataname],
                          axis=1)
            return df
    
    def fisherSmooth(self, n:int=10, alpha:float=0.33, eman:int=1,
                     dataname:str='FS', new_df:pd.DataFrame=None
                     ) -> pd.DataFrame:

        '''
        Calculates the Fisher Transform indicator.

        Parameters
        ----------
        n: int
            Length of the fisher transform.
        alpha: float
            Multiplier for the value calculation.
        eman:int
            Length of the moving average.
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is FT.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the FT.
        '''

        df = self._newDf(new_df, overwrite=False)

        df = self.fisher(n=n,alpha=alpha,dataname='TempFisher')
        df = self.movingAverage(n=eman, method='e', 
                                datatype='TempFisher', 
                                dataname=dataname, new_df=df)

        # Calculamos los niveles del indicador de forma automatica
        cumsum = abs(df[dataname]).cumsum().tolist()
        cmean = cumsum[-1]/len(cumsum)
        maxlevel = cmean
        minlevel = -cmean
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            self.ohlc_df[dataname+'Signal'] = df[dataname+'Signal']
            return self.ohlc_df, maxlevel, minlevel
        
        else:
            df = df.drop(['TempFisher','AbsFS'], axis=1)
            return df, maxlevel, minlevel

    def fisherModified(self,n:int=5,datatype:str='Close',
                       dataname:str='ModFisher', 
                       new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Fisher Modified indicator.

        Parameters
        ----------
        n: int
            Length of the dochian channel.
        datatype: str
            Name of the DataFrame column to use for the indicator 
            calculation.
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is FT.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the FT.
        '''

        df = self._newDf(new_df, overwrite=False)

        # Calculamos el indicador estocastico
        df['mf'] = ((df[datatype] - df['Low'].rolling(n).min()) / \
            (df['High'].rolling(n).max() - df['Low'].rolling(n).min()))
        
        # Primer paso
        df['mf'] = 2 * df['mf'] - 1
        # Filtramos los valores 1 y -1
        df['mf'] = np.where(df['mf'] >= 1, 0.9999, 
                    np.where(df['mf'] <= -1, -0.9999, df['mf']))
        # Aplicamos la transformada de fisher
        df[dataname] = 0.5 * np.log((1+df['mf']) / (1-df['mf']))
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['mf'], axis=1)
            return df

    def trendIntensity(self, n:int=20, method:str='e', datatype:str='Close',
                       dataname:str='TI', new_df:pd.DataFrame=None
                       ) -> pd.DataFrame:

        '''
        Calculates the Trend Intensity indicator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column containing the indicator values.
            Default is TI.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.
            
        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the TI.
        '''

        df = self._newDf(new_df, overwrite=False)

        df = self.movingAverage(n=n, method=method, 
                                          datatype=datatype, 
                                          dataname='TempMA', new_df=df)
        df['Dev'] = np.where(df[datatype] - df['TempMA'] > 0, 1, 0)
        df['Count'] = df['Dev'].rolling(n, min_periods=1).sum()
        df[dataname] = df['Count']/n *100 # + (100 - self.ohlc_df['Count']/n *100)/2
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['TempMA', 'Dev', 'Count'], axis=1)
            return df

    def chandeMomentumOscillator(self, n:int=14, datatype:str='Close', 
                                 dataname:str='CMO', new_df:pd.DataFrame=None
                                ) -> pd.DataFrame:

        '''
        Calculates the Chande Momentum Oscillator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column containing the indicator values.
            Default is CMO.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.
            
        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the CMO.
        '''

        df = self._newDf(new_df, overwrite=False)

        df['Higher'] = np.where(df[datatype] > df[datatype].shift(1), 
                                df[datatype] - df[datatype].shift(1), 0)
        df['Lower'] = np.where(df[datatype] < df[datatype].shift(1), 
                                df[datatype].shift(1) - df[datatype], 0)
        df['Higher'] = df['Higher'].rolling(n).sum()
        df['Lower'] = df['Lower'].rolling(n).sum()
        df[dataname] = (df['Higher'] - df['Lower']) / (df['Higher'] + df['Lower'])
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['Higher', 'Lower'],axis=1)
            return df

    def mbfxTiming(self, n:int=7, fil:int=0, dataname:str='MBFX', 
                   new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the MFBX timing indicator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        fil: int
            Number to add to the indicator.
        dataname: str
            Name of the resulting column containing the indicator values.
            Default is MBFX.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.
            
        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the MBFX.
        '''

        df = self._newDf(new_df, overwrite=False)

        oscilator = [0]*len(df)
        downmove = [0]*len(df)
        upmove = [0]*len(df)
        ld_0, ld_8, ld_16, ld_24, ld_32, ld_40 = 0, 0, 0, 0, 0, 0
        ld_48, ld_56, ld_64, ld_72, ld_80, ld_88 = 0, 0, 0, 0, 0, 0
        ld_96, ld_104, ld_112, ld_120, ld_128, ld_136 = 0, 0, 0, 0, 0, 0
        ld_144, ld_152, ld_160, ld_168, ld_176 = 0, 0, 0, 0, 0
        ld_184, ld_192, ld_200, ld_208 = 0, 0, 0, 0

        index = n-1
        while index <= len(df)-1:
            
            if ld_8 == 0:
                ld_8 = 1.0
                ld_16 = 0.0
                if n-1 >= 5:
                    ld_0 = n-1.0
                else:
                    ld_0 = 5.0
                ld_80 = 100.0 * ((df['High'].tolist()[index] + \
                        df['Low'].tolist()[index] + \
                        df['Close'].tolist()[index]) / 3.0)
                ld_96 = 3.0 / (n + 2.0)
                ld_104 = 1.0 - ld_96
            else:
                if ld_0 <= ld_8:
                    ld_8 = ld_0 + 1.0
                else:
                    ld_8 += 1.0
                ld_88 = ld_80
                ld_80 = 100.0 * ((df['High'].tolist()[index] + \
                        df['Low'].tolist()[index] + \
                        df['Close'].tolist()[index]) / 3.0)
                ld_32 = ld_80 - ld_88
                ld_112 = ld_104 * ld_112 + ld_96 * ld_32
                ld_120 = ld_96 * ld_112 + ld_104 * ld_120
                ld_40 = 1.5 * ld_112 - ld_120 / 2.0
                ld_128 = ld_104 * ld_128 + ld_96 * ld_40
                ld_208 = ld_96 * ld_128 + ld_104 * ld_208
                ld_48 = 1.5 * ld_128 - ld_208 / 2.0
                ld_136 = ld_104 * ld_136 + ld_96 * ld_48
                ld_152 = ld_96 * ld_136 + ld_104 * ld_152
                ld_56 = 1.5 * ld_136 - ld_152 / 2.0
                ld_160 = ld_104 * ld_160 + ld_96 * abs(ld_32)
                ld_168 = ld_96 * ld_160 + ld_104 * ld_168
                ld_64 = 1.5 * ld_160 - ld_168 / 2.0
                ld_176 = ld_104 * ld_176 + ld_96 * ld_64
                ld_184 = ld_96 * ld_176 + ld_104 * ld_184
                ld_144 = 1.5 * ld_176 - ld_184 / 2.0
                ld_192 = ld_104 * ld_192 + ld_96 * ld_144
                ld_200 = ld_96 * ld_192 + ld_104 * ld_200
                ld_72 = 1.5 * ld_192 - ld_200 / 2.0
                if ld_0 >= ld_8 and ld_80 != ld_88:
                    ld_16 = 1.0
                if ld_0 == ld_8 and ld_16 == 0.0:
                    ld_8 = 0.0

            if ld_0 < ld_8 and ld_72 > 0.0000000001:
                ld_24 = 50.0 * (ld_56 / ld_72 + 1.0)
                if ld_24 > 100.0:
                    ld_24 = 100.0
                if ld_24 < 0.0:
                    ld_24 = 0.0
            else:
                ld_24 = 50.0

            oscilator[index] = ld_24
            downmove[index] = ld_24
            upmove[index] = ld_24

            if oscilator[index] < oscilator[index - 1] - fil:
                upmove[index] = np.nan
            elif oscilator[index] > oscilator[index - 1] + fil:
                downmove[index] = np.nan
            else:
                downmove[index] = np.nan
                upmove[index] = np.nan
            
            index += 1

        df[dataname+'MBFXup'] = upmove
        df[dataname+'MBFXdn'] = downmove
        df[dataname] = oscilator
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            self.ohlc_df[dataname+'up'] = df[dataname+'up']
            self.ohlc_df[dataname+'dn'] = df[dataname+'dn']
            return self.ohlc_df
        
        else:
            df = df.drop(['TempMA', 'Dev', 'Count'], axis=1)
            return df

    def shadows(self, n:int=20, method:str='e', dataname:str='Shadow', 
                new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the candle shadows difference and its average.

        Parameters
        ----------
        n: int
            Length of the moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        dataname: str
            Name of the resulting columns containing the indicator values.
            Default is Shadow. It will add a suffix for each column:
            dataname + 'Diff' and dataname + 'DiffMA'.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.
            
        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the indicator columns.
        '''

        df = self._newDf(new_df, overwrite=False)
        
        df[dataname+'UP'] = np.where((df['Close'] > df['Open']), 
                                  df['High'] - df['Close'],
                                  df['High'] - df['Open'])
        df[dataname+'DN'] = np.where((df['Close'] > df['Open']), 
                                    df['Open'] - df['Low'], 
                                    df['Close'] - df['Low'])
        df[dataname+'Diff'] = df[dataname+'UP'] - df[dataname+'DN']
        
        df = self.movingAverage(n=n, method=method, 
                                datatype=dataname+'Diff', 
                                dataname=dataname+'DiffMA', 
                                new_df=df)
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop([dataname+'UP',dataname+'DN'],axis=1)
            return df
    
    def bodySize(self, n:int=20, method:str='e', dataname:str='Body', 
                new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the candle body range and its average.

        Parameters
        ----------
        n: int
            Length of the moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns containing the indicator values.
            Default is Shadow. It will add a suffix for each column:
            dataname + 'Diff' and dataname + 'DiffMA'.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.
            
        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the indicator columns.
        '''

        df = self._newDf(new_df, overwrite=False)

        df[dataname+'UP'] = np.where((df['Close'] > df['Open']), 
                                df['Close'] - df['Open'], 0)
        df[dataname+'DN'] = np.where((df['Close'] < df['Open']), 
                                  df['Open'] - df['Close'], 0)
        
        df = self.movingAverage(n=n, method=method, datatype=dataname+'UP', 
                                dataname=dataname+'UPMA', new_df=df)
        df = self.movingAverage(n=n, method=method, datatype=dataname+'DN', 
                                dataname=dataname+'DNMA', new_df=df)
        df[dataname] = df[dataname+'UPMA'] - df[dataname+'DNMA']

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop([dataname+'UPMA',dataname+'DNMA'],axis=1)
            return df
    
    def demarker(self, n:int=14, method:str='s', dataname:str='DeMark', 
                 new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the DeMarker Oscillator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        dataname: str
            Name of the resulting column containing the indicator values.
            Default is DeMark.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.
            
        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the DeMark.
        '''

        df = self._newDf(new_df, overwrite=False)

        df['DeMAX'] = np.where(df['High'] > df['High'].shift(1), 
                               df['High'] - df['High'].shift(1), 0)
        df['DeMIN'] = np.where(df['Low'] < df['Low'].shift(1), 
                               df['High'].shift(1) - df['High'], 0)
        
        df = self.movingAverage(n=n, method=method, datatype='DeMAX', 
                                dataname='DMaxMAtemp', new_df=df)
        df = self.movingAverage(n=n, method=method, datatype='DeMIN', 
                                dataname='DMinMAtemp', new_df=df)
        
        df[dataname] = df['DMaxMAtemp'] / (df['DMaxMAtemp'] + df['DMinMAtemp'])
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['DeMAX', 'DeMIN', 'DMaxMAtemp', 'DMinMAtemp'],axis=1)
            return df

    def detrendedOscillator(self, n:int=10, method:str='s', datatype:str='Close',
                            dataname:str='DeTrend', new_df:pd.DataFrame=None
                            ) -> pd.DataFrame:

        '''
        Calculates the Detrended Oscillator indicator.

        Parameters
        ----------
        n: int
            Length of the moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is DeTrend.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the DeTrend.
        '''

        df = self._newDf(new_df, overwrite=False)
        
        df = self.movingAverage(n=n, method=method, datatype=datatype, 
                                dataname='DeTrendMA', new_df=df)
        
        df[dataname] = df[datatype].shift(int(n/2+1)) - df['DeTrendMA']
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop(['DeTrendMA'],axis=1)
            return df

    def directionalProbOscillator(self, n:int=10, dataname:str='DProb', 
                                  new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Directional Probability Oscillator indicator.

        Parameters
        ----------
        n: int
            Length of the moving average.
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is DProb.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the DProb.
        '''

        df = self._newDf(new_df, overwrite=False)
            
        df[dataname] = np.where(df['Close'] > df['Open'], 1, 0)
        df[dataname] = df[dataname].rolling(n).sum()
        df[dataname] = df[dataname]/n 
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            return df

    def sar(self, af:float=0.02, amax:float=0.2, dataname:str='PSAR', 
            new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the SAR indicator.

        Parameters
        ----------
        af: float
            Acceleration factor.
        amax float
            Maximum acceleration factor.
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is DProb.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the PSAR.
        '''

        df = self._newDf(new_df, overwrite=False)

        # Starting values
        high = df['High']
        low = df['Low']
        sig0, xpt0, af0 = True, high[0], af
        sar = [low[0] - (high - low).std()]
        for i in range(1, len(df)):
            sig1, xpt1, af1 = sig0, xpt0, af0
            lmin = min(low[i - 1], low[i])
            lmax = max(high[i - 1], high[i])
            if sig1:
                sig0 = low[i] > sar[-1]
                xpt0 = max(lmax, xpt1)
            else:
                sig0 = high[i] >= sar[-1]
                xpt0 = min(lmin, xpt1)
            if sig0 == sig1:
                sari = sar[-1] + (xpt1 - sar[-1])*af1
                af0 = min(amax, af1 + af)
                if sig0:
                    af0 = af0 if xpt0 > xpt1 else af1
                    sari = min(sari, lmin)
                else:
                    af0 = af0 if xpt0 < xpt1 else af1
                    sari = max(sari, lmax)
            else:
                af0 = af
                sari = xpt0
            sar.append(sari)      

        df[dataname] = sar  
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = sar
            return self.ohlc_df
        
        else:
            return df

    def relativeVigorOscillator(self, n:int=10, method:str='s', dataname:str='RVI', 
                                  new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Relative Vigor Index Oscillator indicator.

        Parameters
        ----------
        n: int
            Length of the indicator.
        method: str
            Calculation method used for the moving averages. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting columns in the DataFrame with the data.
            Default is RVI.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the RVI and the RVISig.
        '''

        df = self._newDf(new_df, overwrite=False)

        df[dataname+'Num'] = (df['Close']-df['Open']) + (2*(df['Close'] - df['Open'].shift(1))) \
                            + (2*(df['Close'] - df['Open'].shift(2))) + (df['Close'] - df['Open'].shift(3))
        df = self.movingAverage(n=n, method=method, datatype=dataname+'Num', 
                                dataname=dataname+'Num', new_df=df)
        df[dataname+'Den'] = (df['High']-df['Low']) + (2*(df['High'] - df['Low'].shift(1))) \
                            + (2*(df['High'] - df['Low'].shift(2))) + (df['High'] - df['Low'].shift(3))
        df = self.movingAverage(n=n, method=method, datatype=dataname+'Den', 
                                dataname=dataname+'Den', new_df=df)

        df[dataname] = df[dataname+'Num'] / df[dataname+'Den']
        df[dataname+'Sig'] = ((df[dataname]) + (2*(df[dataname].shift(1))) \
                            + (2*(df[dataname].shift(2))) + (df[dataname].shift(3))) / 6

        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            self.ohlc_df[dataname+'Sig'] = df[dataname+'Sig']
            return self.ohlc_df
        
        else:
            df.drop(columns=[dataname+'Num', dataname+'Den'], inplace=True)
            return df

    # Funcion para medir el rango de vela que se superpone
    def rangeOverlay(self,n=20):
        '''
        Funcion para medir el rango de vela que se superpone
        '''
        
        self.df = self.ohlc_df.copy()
        self.df['OverRange'] = np.where(
                                (self.df['High'] > self.df['High'].shift(1)) & \
                                (self.df['Low'] > self.df['Low'].shift(1)),
                                self.df['High'].shift(1)-self.df['Low'], 0)
        self.df['OverRange'] = np.where(np.logical_and(self.df['High'] < self.df['High'].shift(1),self.df['Low'] < self.df['Low'].shift(1)),self.df['High']-self.df['Low'].shift(1),self.df['OverRange'])
        self.df['OverRange'] = np.where(np.logical_and(self.df['High'] > self.df['High'].shift(1),self.df['Low'] < self.df['Low'].shift(1)),self.df['High'].shift(1)-self.df['Low'].shift(1),self.df['OverRange'])
        self.df['OverRange'] = np.where(np.logical_and(self.df['High'] < self.df['High'].shift(1),self.df['Low'] > self.df['Low'].shift(1)),self.df['High']-self.df['Low'],self.df['OverRange'])
        self.df['OverRange'] = np.where((self.df['Low'] > self.df['High'].shift(1)),0,self.df['OverRange'])
        self.df['OverRange'] = np.where((self.df['High'] < self.df['Low'].shift(1)),0,self.df['OverRange'])
        self.ohlc_df['OR%'] = self.df['OverRange']/(self.df['High'].shift(1)-self.df['Low'].shift(1))*100
        self.ohlc_df['MOR%'] = self.ohlc_df['OR%'].rolling(n).mean()

        return self.ohlc_df

    # Funcion para calcular la media del tamaño de los cuerpos de las velas
    def bodyMa(self,n=20):
        '''
        Funcion para calcular la media del tamaño de los cuerpos de las velas
        '''
        self.df = self.ohlc_df.copy()
        self.df['Body'] = abs(self.df['Close']-self.df['Open'])
        self.df['UpBody'] = np.where(self.df['Close'] > self.df['Open'],self.df['Close']-self.df['Open'],0)
        self.df['DownBody'] = np.where(self.df['Open'] > self.df['Close'],self.df['Open']-self.df['Close'],0)
        self.ohlc_df = self.df
        self.ohlc_df = self.ma(n,datatype='Body',dataname='BodyMA') # Calculamos la media
        self.ohlc_df = self.ma(n,datatype='UpBody',dataname='UpBodyMA')
        self.ohlc_df = self.ma(n,datatype='DownBody',dataname='DownBodyMA')
        self.ohlc_df['BodyDiff'] = self.ohlc_df['UpBodyMA'] - self.ohlc_df['DownBodyMA']
        self.ohlc_df = self.ohlc_df.drop(['Body','UpBody','DownBody'],axis=1)

        return self.ohlc_df

    def momentumBands(self, n:int=20, method:str='s', std:float=2, 
                      datatype:str='Close', dataname:str='MB', 
                      new_df:pd.DataFrame=None) -> pd.DataFrame:

        '''
        Calculates the Momentum Bands indicator.

        Parameters
        ----------
        n: int
            Length of the moving average.
        method: str
            Calculation method used for the moving average. It can be:
            - Simple: s (default)
            - Exponential: e
            - Weighted: w
            - Volume Weighted: v
            - VWAP: vwap
            - Fibonacci: f
        std: float
            Standard Deviation multiplier.
        datatype: str
            Column name to which apply the indicator. Default is Close.
        dataname: str
            Name of the resulting column in the DataFrame with the data.
            Default is MB. There will be three final columns named:
            dataname + 'UP' and dataname + 'DN'.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the MB.
        '''

        df = self._newDf(new_df, overwrite=False)
        
        df['Returns'] = df[datatype] / df[datatype].shift(1) - 1
        df = self.movingAverage(n=n, method=method, datatype='Returns', 
                                dataname=dataname+'MA', new_df=df)
        df[dataname+'STD'] = df['Returns'].rolling(n).std()
        df[dataname+'UP'] = df[datatype].shift(1) * \
                    (1 + (df[dataname+'MA'] + df[dataname+'STD'] * std))
        df[dataname+'DN'] = df[datatype].shift(1) * \
                    (1 + (df[dataname+'MA'] - df[dataname+'STD'] * std))
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname] = df[dataname]
            return self.ohlc_df
        
        else:
            df = df.drop([dataname+'MA',dataname+'STD',
                          'Returns'],axis=1)
            return df

    # Funcion para calcular Ichimoku Cloud
    def ichimoku(self,a=9,b=26,c=52,dataname='Close'):

        '''
        Funcion encargada de calcular la nube de Ichimoku
        '''

        self.ohlc_df['kenkan'] = (self.ohlc_df[dataname].rolling(a).max()+self.ohlc_df[dataname].rolling(a).min())/2
        self.ohlc_df['kijun'] = (self.ohlc_df[dataname].rolling(b).max()+self.ohlc_df[dataname].rolling(b).min())/2
        self.ohlc_df['senkouA'] = ((self.ohlc_df['kenkan']+self.ohlc_df['kijun'])/2).shift(26)
        self.ohlc_df['senkouB'] = ((self.ohlc_df[dataname].rolling(c).max()+self.ohlc_df[dataname].rolling(c).min())/2).shift(26)
        self.ohlc_df['chikou'] = self.ohlc_df[dataname].shift(-26)
        self.ohlc_df['kumo'] = (self.ohlc_df['kenkan']+self.ohlc_df['kijun'])/2-(self.ohlc_df[dataname].rolling(c).max()+self.ohlc_df[dataname].rolling(c).min())/2

        return self.ohlc_df

    def riccochet(self, window1:int=10, window2:int=20, datatype='Close', 
                  dataname:str='R', new_df:pd.DataFrame=None
                  ) -> pd.DataFrame:

        '''
        Calculates the Riccochet Bollinger Bands.

        Parameters
        ----------
        window1: int
            Length of the first moving average.
        window2: int
            Length of the second moving average.
        datatype: str
            Name of the column to use for the moving averages.
        dataname: str
            Start of the columns returned with the calculates data.
        new_df: pd.DataFrame
            DataFrame to use in case you don't want to use the object data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            Contains all the DataFrame data plus the MB.
        '''

        df = self._newDf(new_df, overwrite=False)

        # Calculate the first set of Bollinger Bands
        df[dataname+'MA1'] = df[datatype].rolling(window=window1).mean()
        df[dataname+'STD1'] = df[datatype].rolling(window=window1).std()
        df[dataname+'UP1'] = df[dataname+'MA1'] + 2 * df[dataname+'STD1']
        df[dataname+'DN1'] = df[dataname+'MA1'] - 2 * df[dataname+'STD1']
        
        # Calculate the second set of Bollinger Bands
        df[dataname+'MA2'] = df[datatype].rolling(window=window2).mean()
        df[dataname+'STD2'] = df[datatype].rolling(window=window2).std()
        df[dataname+'UP2'] = df[dataname+'MA2'] + 2 * df[dataname+'STD2']
        df[dataname+'DN2'] = df[dataname+'MA2'] - 2 * df[dataname+'STD2']
        
        if not isinstance(new_df, pd.DataFrame):
            self.ohlc_df[dataname+'UP1'] = df[dataname+'UP1']
            self.ohlc_df[dataname+'DN1'] = df[dataname+'DN1']
            self.ohlc_df[dataname+'UP2'] = df[dataname+'UP2']
            self.ohlc_df[dataname+'DN2'] = df[dataname+'DN2']
            return self.ohlc_df
        
        else:
            df.drop(columns=[dataname+'MA1',dataname+'STD1',
                             dataname+'MA2', dataname+'STD2'], 
                             inplace=True)
            return df

    def doubleRiccochet(self, df:pd.DataFrame=None, window1:int=10, window2:int=20, 
                        offset:int=1, delay:int=1, inverse:bool=False, 
                        new_df:pd.DataFrame=None) -> pd.DataFrame:

        df = self._newDf(new_df, overwrite=False)

        df = self.riccochet(df, window1, window2, drop=False, new_df=new_df)

        if delay + offset < 0 or delay < 0:
            raise ValueError('You are trying to look in the future!')

        touch_up_band1 = (df['High'].shift(delay) > df['RUP1'].shift(delay)) & \
                (df['High'].shift(delay+offset) <= df['RUP1'].shift(delay+offset))
        touch_dn_band1 = (df['Low'].shift(delay) < df['RDN1'].shift(delay)) & \
                (df['Low'].shift(delay+offset) >= df['RDN1'].shift(delay+offset))

        touch_up_band2 = (df['High'].shift(delay) > df['RUP2'].shift(delay)) & \
                (df['High'].shift(delay+offset) <= df['RUP2'].shift(delay+offset))
        touch_dn_band2 = (df['Low'].shift(delay) < df['RDN2'].shift(delay)) & \
                (df['Low'].shift(delay+offset) >= df['RDN2'].shift(delay+offset))

        buy_cond = touch_up_band1.shift(1) & touch_dn_band2
        sell_cond = touch_dn_band1.shift(1) & touch_up_band2

        if inverse:
            df['DRicco'] = np.where(buy_cond, -1, 
                        np.where(sell_cond, 1, 0))
        else:
            df['DRicco'] = np.where(buy_cond, 1, 
                        np.where(sell_cond, -1, 0))
        
        df['Result'] = np.where(df['DRicco'] > 0, df['Close'].shift(-4) - df['Close'],
                    np.where(df['DRicco'] < 0, df['Close'] - df['Close'].shift(-4), 0))

        return df




class Statistics(OHLC):

    '''
    Class used to calculate statistical indicators.
    '''

    def __init__(self, ohlc:pd.DataFrame=None, errors:bool=True, 
                 verbose:bool=True) -> None:

        '''
        Function initiate the Indicators class.

        Parameters
        ----------
        ohlc: pd.DataFrame
            DataFrame with OHCL data for an asset. The open columns must 
            be named 'Open', the close column 'Close', the high column
            'High' and the low column 'Low
        
        '''

        self.errors = errors
        self.verbose = verbose
        self._newDf(ohlc, needed_cols=['Open', 'High', 'Low', 'Close'], 
                    overwrite=True)

    def mean(self, n:int=None, datatype:str='Close', 
             dataname:str='Mean') -> pd.DataFrame:

        '''
        Calculates the mean of a column in a dataframe.

        Parameters
        ----------
        n: int
            Length of the moving average.
        datatype: str
            Name of the column to calculate the average with.
        dataname: str
            Name of the column to store the average data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame of the object containing the new average data.
        '''

        df = self.ohlc_df.copy()
        
        if n == None or n > len(df):
            n = len(df)
        
        if n <= 0:
            raise ValueError('mean requires at least one data point')
        
        df[dataname] = df[datatype].rolling(n).mean()

        self.ohlc_df[dataname] = df[dataname]

        return self.ohlc_df

    def median(self, n:int=None, datatype:str='Close', 
             dataname:str='Median') -> pd.DataFrame:

        '''
        Calculates the median of a column in a dataframe.

        Parameters
        ----------
        n: int
            Length of the moving average.
        datatype: str
            Name of the column to calculate the average with.
        dataname: str
            Name of the column to store the average data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame of the object containing the new average data.
        '''

        df = self.ohlc_df.copy()
        
        if n == None or n > len(df):
            n = len(df)
        
        if n <= 0:
            raise ValueError('mean requires at least one data point')
        
        df[dataname] = df[datatype].rolling(n).median()

        self.ohlc_df[dataname] = df[dataname]

        return self.ohlc_df

    def mode(self, n:int=None, datatype:str='Close', 
             dataname:str='Mode', average:bool=False) -> pd.DataFrame:

        '''
        Calculates the mode of a column in a dataframe.

        Parameters
        ----------
        n: int
            Length of the moving mode.
        datatype: str
            Name of the column to calculate the mode with.
        dataname: str
            Name of the column to store the mode data.
        average: bool
            True to calculate the mean in case there are multiple 
            modes.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame of the object containing the new mode data.
        '''

        df = self.ohlc_df.copy()
        
        if n == None or n > len(df):
            n = len(df)
        
        if n <= 0:
            raise ValueError('mean requires at least one data point')
        
        def _mean(l:list):

            return sum(l)/len(l)
        
        def _mode(l:list):

            l.sort()
            
            l2 = []
            for i in l:
                l2.append(l.count(i))
                
            d1 = dict(zip(l, l2))
            
            mode = [k for (k,v) in d1.items() if v == max(l2)]
            if average and len(mode) > 1:
                mode = [_mean(mode)]
            
            return mode

        df[dataname] = df[datatype].rolling(n).apply(lambda x: _mode(x.tolist()))

        self.ohlc_df[dataname] = df[dataname]

        return self.ohlc_df

    def squareDeviation(self, n:int=None, datatype:str='Close', 
             dataname:str='Median') -> pd.DataFrame:

        '''
        Return square deviations of a column in a dataframe.

        Parameters
        ----------
        n: int
            Length of the moving Square Deviation.
        datatype: str
            Name of the column to calculate the Square 
            Deviation with.
        dataname: str
            Name of the column to store the Square 
            Deviation data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame of the object containing the new Square 
            Deviation data.
        '''

        df = self.ohlc_df.copy()
        
        if n == None or n > len(df):
            n = len(df)
        
        if n <= 0:
            raise ValueError('mean requires at least one data point')
        
        df = self.mean(n=n, datatype=datatype, dataname='TempMean').copy()
        df[dataname] = ((df[datatype]-df['TempMean'])**2)

        self.ohlc_df[dataname] = df[dataname]

        return self.ohlc_df

    def standardDeviation(self, n:int=None, datatype:str='Close', 
             dataname:str='Median') -> pd.DataFrame:

        '''
        Calculates the Standard Deviation of a column in a 
        dataframe.

        Parameters
        ----------
        n: int
            Length of the moving Standard Deviation.
        datatype: str
            Name of the column to calculate the Standard Deviation 
            with.
        dataname: str
            Name of the column to store the Standard Deviation 
            data.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame of the object containing the new Standard 
            Deviation data.
        '''

        df = self.ohlc_df.copy()
        
        if n == None or n > len(df):
            n = len(df)
        
        if n <= 0:
            raise ValueError('mean requires at least one data point')

        if n < 2:
            raise ValueError('variance requires at least two data points')
        
        df[dataname] = df[datatype].rolling(n).std()

        self.ohlc_df[dataname] = df[dataname]

        return self.ohlc_df

    



class CandlePatterns(OHLC):

    '''
    Clase con las funciones de reconocimiento de patrones
    '''

    def __init__(self, ohlc:pd.DataFrame=None, errors:bool=True, 
                 verbose:bool=True, volume:bool=False) -> None:

        '''
        Function initiate the Indicators class.

        Parameters
        ----------
        ohlc: pd.DataFrame
            DataFrame with OHCL data for an asset. The open columns must 
            be named 'Open', the close column 'Close', the high column
            'High' and the low column 'Low
        
        '''

        self.errors = errors
        self.verbose = verbose
        self._newDf(ohlc, overwrite=True)
        self.volume = volume

    def isDoji(self, dataname:str='Doji', offset:int=1) -> pd.DataFrame:

        '''
        Detects Bullish and Bearish Doji candles.

        Parameters
        ----------
        dataname: str
            Name of the column to store the new data.
        offset: int
            Candles offset to detect.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame containing the new data.
        '''

        df = self.ohlc_df.copy()
        bullish_cond = (df['Close'].shift(1) < df['Open'].shift(1)) & \
                        (df['Close'] == df['Open'])
        bearish_cond = (df['Close'].shift(1) > df['Open'].shift(1)) & \
                        (df['Close'] == df['Open'])
        
        self.olhc_df[dataname] = np.where(bullish_cond, 1, 
                            np.where(bearish_cond, -1, 0)).shift(offset)

        return self.ohlc_df

    def isHarami(self, dataname:str='Harami', offset:int=1) -> pd.DataFrame:

        '''
        Detects Bullish and Bearish Harami candles.

        Parameters
        ----------
        dataname: str
            Name of the column to store the new data.
        offset: int
            Candles offset to detect.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame containing the new data.
        '''

        df = self.ohlc_df.copy()
        bullish_cond = (df['Close'].shift(1) < df['Open'].shift(1)) & \
                        (df['Close'] > df['Open']) & \
                        (df['Close'].shift(1) < df['Low']) & \
                        (df['Open'].shift(1) > df['High'])
        bearish_cond = (df['Close'].shift(1) > df['Open'].shift(1)) & \
                        (df['Close'] < df['Open']) & \
                        (df['Close'].shift(1) > df['High']) & \
                        (df['Open'].shift(1) < df['Low'])
        
        self.olhc_df[dataname] = np.where(bullish_cond, 1, 
                            np.where(bearish_cond, -1, 0)).shift(offset)

        return self.ohlc_df

    def isEngulfing(self, dataname:str='Harami', offset:int=1) -> pd.DataFrame:

        '''
        Detects Bullish and Bearish Engulfing candles.

        Parameters
        ----------
        dataname: str
            Name of the column to store the new data.
        offset: int
            Candles offset to detect.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame containing the new data.
        '''

        df = self.ohlc_df.copy()
        bullish_cond = (df['Close'].shift(1) < df['Open'].shift(1)) & \
                        (df['Close'] > df['Open']) & \
                        (df['Open'].shift(1) < df['Close'])
        bearish_cond = (df['Close'].shift(1) > df['Open'].shift(1)) & \
                        (df['Close'] < df['Open']) & \
                        (df['Open'].shift(1) > df['Close'])
        
        self.olhc_df[dataname] = np.where(bullish_cond, 1, 
                            np.where(bearish_cond, -1, 0)).shift(offset)

        return self.ohlc_df

    def isPiercing(self, dataname:str='Harami', offset:int=1) -> pd.DataFrame:

        '''
        Detects Bullish and Bearish Piercing candles.

        Parameters
        ----------
        dataname: str
            Name of the column to store the new data.
        offset: int
            Candles offset to detect.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame containing the new data.
        '''

        df = self.ohlc_df.copy()
        bullish_cond = (df['Close'].shift(1) < df['Open'].shift(1)) & \
                        (df['Close'] > df['Open']) & \
                        (df['Open'] < df['Close'].shift(1)) & \
                        (df['Close'] > df['Close'].shift(1)) & \
                        (df['Close'] < df['Open'].shift(1))
        bearish_cond = (df['Close'].shift(1) > df['Open'].shift(1)) & \
                        (df['Close'] < df['Open']) & \
                        (df['Open'] > df['Close'].shift(1)) & \
                        (df['Close'] < df['Close'].shift(1)) & \
                        (df['Close'] > df['Open'].shift(1))
        
        self.olhc_df[dataname] = np.where(bullish_cond, 1, 
                            np.where(bearish_cond, -1, 0)).shift(offset)

        return self.ohlc_df

    def isPinbar(self, dataname:str='Pinbar', offset:int=1, 
                 range_mult:float=0.34) -> pd.DataFrame:

        '''
        Detects Bullish and Bearish Pinbar candles.

        Parameters
        ----------
        dataname: str
            Name of the column to store the new data.
        offset: int
            Candles offset to detect.
        range_mult: float
            High-Low range fraction which can't be excedeed by the 
            open-close range.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame containing the new data.
        '''

        df = self.ohlc_df.copy()
        bullish_cond = (df['Close'].shift(1) < df['Open'].shift(1)) & (df['Close'] > df['Open']) & \
                        (df['Open'] > (2*df['High']+df['Low']) / 3) & \
                        (df['Close'] > (2*df['High']+df['Low']) / 3) & \
                        (df['Close']-df['Open'] <= range_mult*(df['High']-df['Low']))
        bearish_cond = (df['Close'].shift(1) > df['Open'].shift(1)) & (df['Close'] < df['Open']) & \
                        (df['Open'] < (df['High']+2*df['Low']) / 3) & \
                        (df['Close'] < (df['High']+2*df['Low']) / 3) & \
                        (df['Close']-df['Open'] <= range_mult*(df['High']-df['Low']))
        
        self.olhc_df[dataname] = np.where(bullish_cond, 1, 
                            np.where(bearish_cond, -1, 0)).shift(offset)

        return self.ohlc_df

    def isT2(self, dataname:str='Pinbar', offset:int=1, 
                 range_mult:float=1.5) -> pd.DataFrame:

        '''
        Detects Bullish and Bearish T2 candles.

        Parameters
        ----------
        dataname: str
            Name of the column to store the new data.
        offset: int
            Candles offset to detect.
        range_mult: float
            Volatility range multiplier which has to be excedeed 
            by the actual range.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame containing the new data.
        '''

        df = self.ohlc_df.copy()
        df['max_range'] = abs(df['High']-df['Low']).rolling(10).max()

        bullish_cond = (df['Close'].shift(1) > df['Open'].shift(1)) & \
                        (df['High']-df['Low'] < (df['High']-df['Low']/2).shift(1)) & \
                        ((df['High']-df['Low']).shift(1) > range_mult*df['max_range'].shift(2)) & \
                        (df['Low'] >= (df['High']+df['Low']).shift(1) / 2)
        bearish_cond = (df['Close'].shift(1) < df['Open'].shift(1)) & \
                        (df['High']-df['Low'] < (df['High']-df['Low']/2).shift(1)) & \
                        ((df['High']-df['Low']).shift(1) > range_mult*df['max_range'].shift(2)) & \
                        (df['High'] >= (df['High']+df['Low']).shift(1) / 2)
        
        self.olhc_df[dataname] = np.where(bullish_cond, 1, 
                            np.where(bearish_cond, -1, 0)).shift(offset)

        return self.ohlc_df

    def is4bar(self, dataname:str='FourBar', offset:int=1, 
                 range_mult:float=1.5) -> pd.DataFrame:

        '''
        Detects Bullish and Bearish 4bars patterns.

        Parameters
        ----------
        dataname: str
            Name of the column to store the new data.
        offset: int
            Candles offset to detect.
        range_mult: float
            Volatility range multiplier which has to be excedeed 
            by the actual range.

        Returns
        -------
        ohlc_df: pd.DataFrame
            DataFrame containing the new data.
        '''

        df = self.ohlc_df.copy()
        df['max_range'] = abs(df['High']-df['Low']).rolling(10).max()

        bullish_cond = (df['Close'].shift(2) > df['Open'].shift(2)) & \
                        ((df['High']-df['Low']).shift(2) > range_mult*df['max_range'].shift(3)) & \
                        ((df['High']-df['Low']).shift(1) < (df['High']-df['Low']/2).shift(2)) & \
                        (df['High']-df['Low'] < (df['High']-df['Low']/2).shift(2)) & \
                        (df['Low'] >= (df['High']+df['Low']).shift(2) / 2) & \
                        (df['Low'].shift(1) >= (df['High']+df['Low']).shift(2) / 2)
        
        bearish_cond = (df['Close'].shift(2) < df['Open'].shift(2)) & \
                        ((df['High']-df['Low']).shift(2) > range_mult*df['max_range'].shift(3)) & \
                        ((df['High']-df['Low']).shift(1) < (df['High']-df['Low']/2).shift(2)) & \
                        (df['High']-df['Low'] < (df['High']-df['Low']/2).shift(2)) & \
                        (df['High'] <= (df['High']+df['Low']).shift(2) / 2) & \
                        (df['High'].shift(1) <= (df['High']+df['Low']).shift(2) / 2)
        
        self.olhc_df[dataname] = np.where(bullish_cond, 1, 
                            np.where(bearish_cond, -1, 0)).shift(offset)

        return self.ohlc_df



# Prueba de los indicadores, solo se ejecuta cuando el archivo se corre directamente
if __name__ == '__main__':

    import matplotlib.pyplot as plt
        
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

    indicator = Indicators(raw)
    
    data = indicator.rsi()
    data = indicator.bollingerBands(n=5, desvi=0.5)

