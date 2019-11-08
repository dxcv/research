import numpy as np
import pandas as pd

from backtesting.WalkForwardBacktest import WalkForwardBtNoCompounding
from backtesting.portfolio_analysis import Comparis, PortfolioResults
from classes.Futures import FuturesData
from classes.portfolio_construction import HRP, ERC
from configs.futures_config import futures_config

# get universe config
futures_data = FuturesData(futures_config)

# get price and return data
futures_prices_dataset = futures_data.get_futures_prices_dataset()
futures_returns = futures_data.get_futures_returns()

# get HRP weights
hrp = HRP.HRP(futures_prices_dataset.near, window_length=125, futures_returns=futures_returns)
hrp.hrp_calculation_through_time_in_parallel()

# ERC
lower_bounds = np.array([0, 0])
upper_bounds = np.array([1, 1])
erc = ERC.RiskParity(futures_prices_dataset.near, window=125,
                     futures_returns=futures_returns.loc[hrp.hrp_weights.index],
                     lower_bound=lower_bounds, upper_bound=upper_bounds)
erc_w = erc.calculate_rolling_erc_allocation()

# hrp bt
bt = WalkForwardBtNoCompounding(hrp.hrp_weights, futures_prices_dataset.near, aum=100)
bt.calculate_backtest_performance()

# get ew hrp
ew = pd.DataFrame(index=hrp.hrp_weights.index, columns=hrp.hrp_weights.columns).fillna(1) / hrp.hrp_weights.shape[1]
bt_ew = WalkForwardBtNoCompounding(ew, futures_prices_dataset.near, aum=100)
bt_ew.calculate_backtest_performance()

# erc bt
bt_erc = WalkForwardBtNoCompounding(erc_w, futures_prices_dataset.near, aum=100)
bt_erc.calculate_backtest_performance()

# results analysis
results = PortfolioResults(bt, name='HRP')
results_ew = PortfolioResults(bt_ew, name='EW')
results_erc = PortfolioResults(bt_erc, name='ERC')
res = Comparis(HRP=results, EW=results_ew, ERC=results_erc)
