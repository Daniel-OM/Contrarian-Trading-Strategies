
import datetime as dt
from backtest import AssetConfig, Commissions, StrategyConfig

trades_url = 'execution' # Directory where to store the files with the trades
open_trades_name = 'open_trades.csv' # Name of the file contianing the open trades
closed_trades_name = 'closed_trades.csv' # Name of the file contianing the closed trades
broker = 'degiro' # Name of the broker from which to retrieve data
execute = False # True to execute orders with broker
apply_filter = True # True to apply MA filter to strategies
start_time = dt.datetime.strptime('09:00', '%H:%M').time() # dt.time(9, 0) # Starting time to check
end_time = dt.datetime.strptime('15:30', '%H:%M').time() # dt.time(15, 30) # Ending time to check

tickers = {
    'SP500': {'yfinance':'500.PA', 'degiro':'LU1681048804'},
    'NASDAQ': {'yfinance':'ANX.PA', 'degiro':'LU1681038243'}, 
    'STOXX': {'yfinance':'C50.PA'}, 
    'MSCI World': {'yfinance':'CW8.PA'}
}

strategies = {
    'detrended': StrategyConfig(name='detrended', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1', filter='MA_100'), # 2
    'envelopes': StrategyConfig(name='envelopes', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'fibTiming': StrategyConfig(name='fibTiming', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'rsiExtremeDuration': StrategyConfig(name='rsiExtremeDuration', assets={
        'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'), # 2
    'chandeMomentum': StrategyConfig(name='chandeMomentum', assets={
        'SP500': AssetConfig(name='SPY', risk=0.04, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1'),
    'macdTrend': StrategyConfig(name='macdTrend', assets={
        'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'rsiAtr': StrategyConfig(name='rsiAtr', assets={
        'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'stochExtreme': StrategyConfig(name='stochExtreme', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'trendContinuation': StrategyConfig(name='trendContinuation', assets={
        'SP500': AssetConfig(name='SPY', risk=0.005, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1', filter='MA_100'),
    'trendInten': StrategyConfig(name='trendInten', assets={
        'SP500': AssetConfig(name='SPY', risk=0.04, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'), # 2
    'turtlesBreakout': StrategyConfig(name='turtlesBreakout', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1', filter='MA_100'),
    'dailyPB': StrategyConfig(name='dailyPB', assets={
        'SP500': AssetConfig(name='SPY', risk=0.015, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1', filter='MA_100'),
    'volatPB': StrategyConfig(name='volatPB', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=3, timeframe='D1'),
    'pullbackBounce': StrategyConfig(name='pullbackBounce', assets={
        'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=2, timeframe='D1'),
    'atrExt': StrategyConfig(name='atrExt', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=10.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=False, time_limit=10, timeframe='D1', filter='MA_100'),
    'kamaTrend': StrategyConfig(name='kamaTrend', assets={
        'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=10.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=10, timeframe='D1'),
    'rsiNeutrality': StrategyConfig(name='rsiNeutrality', assets={
        'SP500': AssetConfig(name='SPY', risk=0.02, sl=2.0, tp=4.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=10, timeframe='D1'),
    'paraSar': StrategyConfig(name='paraSar', assets={
        'SP500': AssetConfig(name='SPY', risk=0.03, sl=2.0, tp=5.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'momentum': StrategyConfig(name='momentum', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=2.0, tp=5.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'adxMomentum': StrategyConfig(name='adxMomentum', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=100, timeframe='D1'),
    'weeklyDip': StrategyConfig(name='weeklyDip', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
    'stochDip': StrategyConfig(name='stochDip', assets={
        'SP500': AssetConfig(name='SPY', risk=0.01, sl=4.0, tp=2.0, order='stop', min_size=1, max_size=5000, 
                             commission=Commissions('perunit', 0.05, cmin=1), 
                             drawdown=None),
    }, use_sl=True, use_tp=True, time_limit=5, timeframe='D1'),
}
