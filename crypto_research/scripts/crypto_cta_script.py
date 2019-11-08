import pandas as pd

from backtesting.WalkForwardBacktest import WalkForwardBtNoCompounding
from backtesting.portfolio_analysis import PortfolioResults, Comparis
from configs.strategy_configs.crypto_cta import crypto_cta_config
from configs.universe_spec import universe_crypto
from data_loading.load_from_disk.load_crypto_data import load_crypto_data_from_disk
from signals.momentum.time_series_momentum import exp_ma_crossover
from utils.splines import spline_series, cta_spline, grd_spline

# load crypto data
crypto = load_crypto_data_from_disk(universe_crypto, frequency='1H')
prices = crypto.get_cross_sectional_view('Close')

# calculate signal
cma = pd.DataFrame()
splined_cma = pd.DataFrame()
splined_cma_grd = pd.DataFrame()

for instrument in prices:
    cma[instrument] = exp_ma_crossover(prices[instrument], 2, 8, 20, vol_floor=0.1)
    splined_cma[instrument] = spline_series(cma[instrument], cta_spline)
    splined_cma_grd[instrument] = spline_series(cma[instrument], grd_spline)

splined_cma /= 2

# backtests
bt_raw = WalkForwardBtNoCompounding(cma, prices, aum=100, **crypto_cta_config)
bt_spline_cta = WalkForwardBtNoCompounding(splined_cma, prices, aum=100,
                                           **crypto_cta_config)
bt_spline_grd = WalkForwardBtNoCompounding(splined_cma_grd, prices, aum=100,
                                           **crypto_cta_config)
bt_raw.calculate_backtest_performance()
bt_spline_grd.calculate_backtest_performance()
bt_spline_cta.calculate_backtest_performance()

# results
res_l = PortfolioResults(bt_spline_grd, 'l_spline')
res_ls = PortfolioResults(bt_spline_cta, 'ls_spline')
res_ls_raw = PortfolioResults(bt_raw, 'ls_raw')

comp = Comparis(LS=res_ls, L=res_l, RAW=res_ls_raw)

comp.create_results_comparison()
comp.plot_cumulative_returns()
