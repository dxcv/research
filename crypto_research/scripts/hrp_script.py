import pandas as pd

from backtesting.WalkForwardBacktest import WalkForwardBtNoCompounding
from classes.portfolio_construction.HRP import HRP
from configs.universe_spec import universe_crypto
from data_loading.load_from_disk.load_crypto_data import load_crypto_data_from_disk

# load crypto data
crypto = load_crypto_data_from_disk(universe_crypto, frequency='1D')
prices = crypto.get_cross_sectional_view('Close')

# create hrp object
hrp = HRP(prices, window_length=30)  # cleaning=True

# calculate HRP weights
hrp.hrp_calculation_through_time_in_parallel()

# plot
ax1 = hrp.hrp_weights.plot.area()

# backtest hrp
bt_hrp = WalkForwardBtNoCompounding(hrp.hrp_weights, prices, aum=100)

# backtest ew
ew = pd.DataFrame(index=hrp.hrp_weights.index, columns=hrp.hrp_weights.columns)
ew[hrp.hrp_weights.notna()] = 1
ew = ew.apply(lambda x: x / x.notna().sum(), axis=1)

bt_ew = WalkForwardBtNoCompounding(ew, prices, aum=100)

# calculate and plot performance
bt_hrp.calculate_backtest_performance()
bt_ew.calculate_backtest_performance()

ax = bt_hrp.backtest.net_asset_value.plot(label='HRP', legend=True)
bt_ew.backtest.net_asset_value.plot(ax=ax, label='EW', legend=True)

# calculate daily turnover
hrp.hrp_weights.diff().abs().sum(axis=1).mean()
# I should really do this with target weights - implied weights at turnover point, ie holdings * price
# (hrp.hrp_weights - (bt_hrp.backtest.holdings.shift(1) * bt_hrp.close)/100).abs().sum(axis=1).mean()
# yesterdays holdings * todays price

