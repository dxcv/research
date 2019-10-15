import datetime as dt

import numpy as np
import pandas as pd

from backtesting.WalkForwardBacktest import WalkForwardBtCompounding
from data_loading.load_from_disk.load_equities_data import load_equities_data_from_disk
from signals.momentum.time_series_momentum import exp_ma_crossover
from utils.splines import spline_series, grd_spline
from utils.resampling_and_risk_metrics import calculate_sample_stats
from configs.universe_spec import universe_tech
from data_loading.query_from_api.query_data_from_iex import query_and_write_eod_data_for_symbols_from_iex

# potentially update data from api source
current_date = dt.date.today().strftime('%Y%m%d')
symbols_list = ['AAPL', 'AMZN', 'FB', 'NFLX', 'GOOG']


# for symbol in symbol_new:
#     query_and_write_eod_data_for_symbols_from_iex([symbol], freq='B', date=current_date, attribute='close')
#     # todo: dynamic updating; read data from disk, check most recent date, potentially update
#     # put this into a separate file


# load crypto data from disk
# crypto_data = load_crypto_data_from_disk(universe_crypto, frequency='1D')
# etf_data = load_etf_data_from_disk(['FXI', 'SPY'], frequency='B')
tech_data = load_equities_data_from_disk(symbols_list, frequency='B')

close_data = tech_data.get_cross_sectional_view('close')
# open_data = tech_data.get_cross_sectional_view('Open')

# btcusd = crypto_data.raw_data.BTCUSD.get_df_view(['Open', 'High', 'Low', 'Close', 'VolumeUSD'])

# calculate expma signal
cma = pd.DataFrame()
splined_cma = pd.DataFrame()

for instrument in close_data:
    cma[instrument] = exp_ma_crossover(close_data[instrument], 2, 8, 20, vol_floor=0.1)
    splined_cma[instrument] = spline_series(cma[instrument], grd_spline)

splined_cma /= 2

# backtest raw
bt_raw = WalkForwardBtCompounding(cma, close_data, 1, **{
    'max_gross_leverage': 2, 'max_net_leverage': 2, 'max_pos_leverage': 1})
bt_raw.calculate_backtest_performance()

# backtest splined
bt_spline = WalkForwardBtCompounding(splined_cma, close_data, aum=10000000,
                                     max_gross_leverage=2, max_net_leverage=2, max_pos_leverage=1)

bt_spline.calculate_backtest_performance()

# backtest constant exposure
bt_ce = WalkForwardBtCompounding(splined_cma, close_data, 1,
                                 max_gross_leverage=1, max_net_leverage=1, max_pos_leverage=1, constant_exposure=True)

bt_ce.calculate_backtest_performance()

# backtest ew
ew_df = pd.DataFrame(np.ones(splined_cma.shape), columns=splined_cma.columns, index=splined_cma.index)

bt_ew = WalkForwardBtCompounding(ew_df, close_data, 1,
                                 max_gross_leverage=1, max_net_leverage=1, max_pos_leverage=1,
                                 constant_exposure=True)

# plot
# bt_raw.backtest.net_asset_value.plot(label='raw', legend=True)
bt_spline.backtest.net_asset_value.plot(label='spline', legend=True)
bt_ce.backtest.net_asset_value.plot(label='ce', legend=True)
bt_ew.backtest.net_asset_value.plot(label='ew', legend=True)

# ew
# ew_df = pd.DataFrame(np.ones(splined_cma.shape), columns=splined_cma.columns, index=splined_cma.index)
#
# bt_ew = WalkForwardBtCompounding(ew_df, close_data, 1,
#                                      max_gross_leverage=1,max_net_leverage=1, max_pos_leverage=1,
#                                      constant_exposure=False)

calculate_sample_stats(bt_spline.backtest.net_asset_value, frequency='B', name='spline')

# pnl attribution
attribution = (bt_spline.holdings * bt_spline.pnl_per_instrument).cumsum().dropna()