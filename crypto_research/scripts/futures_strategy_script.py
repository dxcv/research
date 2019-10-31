from classes.Futures import FuturesData
from configs.futures_config import futures_config
from classes.portfolio_construction import HRP
from backtesting.WalkForwardBacktest import WalkForwardBtNoCompounding
from backtesting.portfolio_analysis import Comparis, PortfolioResults
import pandas as pd

# get universe config
futures_data = FuturesData(futures_config)

# get price and return data
futures_prices_dataset = futures_data.get_futures_prices_dataset()
futures_returns = futures_data.get_futures_returns()

# get HRP weights
hrp = HRP.HRP(futures_prices_dataset.near, window_length=125, futures_returns=futures_returns)
hrp.hrp_calculation_through_time_in_parallel()

# hrp bt
bt = WalkForwardBtNoCompounding(hrp.hrp_weights, futures_prices_dataset.near, aum=100)
bt.calculate_backtest_performance()


# get ew hrp
ew = pd.DataFrame(index=hrp.hrp_weights.index, columns=hrp.hrp_weights.columns).fillna(1)/hrp.hrp_weights.shape[1]
bt_ew = WalkForwardBtNoCompounding(ew, futures_prices_dataset.near, aum=100)
bt_ew.calculate_backtest_performance()

# results analysis
results = PortfolioResults(bt, name='HRP')
results_ew = PortfolioResults(bt_ew, name='EW')
res = Comparis(HRP=results, EW=results_ew)

