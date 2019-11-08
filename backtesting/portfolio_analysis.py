import numpy as np
import pandas as pd
from utils.dataset import DataSet


class PortfolioResults(object):
    """
    initiate with a strategy (WalkForwardBt) object
    should I add start and end dates for each individual calculation?
    proper frequency and re-balancing handling for each calculation (incl. base case monthly data)
    """

    def __init__(self, strategy, name='default_strategy', date=None):
        # todo: way to handle a variable start date: needed in particular for Comparis
        self.net_asset_value = strategy.backtest.net_asset_value
        self.holdings = strategy.backtest.holdings
        self.instrument_prices = strategy.close
        self.portfolio_weights = strategy.scaled_weights
        self.frequency = strategy.frequency
        if date is not None:
            self.net_asset_value = strategy.backtest.net_asset_value.loc[date:]
            self.holdings = strategy.backtest.holdings.loc[date:]
            self.instrument_prices = strategy.close.loc[date:]
            self.portfolio_weights = strategy.scaled_weights.loc[date:]
            self.frequency = strategy.frequency

        self.name = name

    def _calculate_turnover_series(self):
        prices = self.instrument_prices.copy()
        holdings = self.holdings.copy()
        weights = self.portfolio_weights.copy()

        daily_turnover = (weights - (holdings.shift(1) * prices) / 100).abs().sum(axis=1)
        return daily_turnover

    def _calculate_returns(self):
        nav = self.net_asset_value.copy()
        returns = np.log(nav).diff(1).dropna()
        return returns

    def _calculate_mean_ann_return(self):
        returns = self._calculate_returns()
        scaling = self._calculate_scaling()

        mean_ann_ret = returns.mean() * scaling
        return mean_ann_ret

    def _calculate_scaling(self):
        frequency = self.frequency
        if frequency == 'D':
            scaling = 365
        elif frequency == 'B':
            scaling = 260
        elif frequency == '1H':
            scaling = 24 * 365
        else:
            assert frequency is None
            scaling = 260
        return scaling

    def _calculate_drawdown_series(self):
        prices = self.net_asset_value
        maxima = prices.expanding().max()
        drawdown = (prices - maxima) / maxima
        return drawdown

    def _calculate_max_drawdown(self):
        dd_series = self._calculate_drawdown_series()
        max_dd = pd.Series(data=dd_series.min(), index=[dd_series.idxmin()])
        return max_dd

    def _calculate_time_under_water(self):
        dd_series = self._calculate_drawdown_series()
        dd_df = pd.DataFrame(dd_series)
        dd_df['not_in_dd'] = dd_series == 0
        time_under_water = dd_df[dd_df['not_in_dd']].index.to_series().diff()  # should I shift this by -1?
        return time_under_water

    def _calculate_max_time_under_water(self):
        tuw = self._calculate_time_under_water()
        max_tuw = pd.Series(data=tuw.max(), index=[tuw.idxmax()])
        return max_tuw

    def _calculate_vol(self):
        returns = self._calculate_returns()
        vol = returns.std()
        return vol

    def _calculate_ann_vol(self):
        vol = self._calculate_vol()
        scaling = self._calculate_scaling()
        return vol * np.sqrt(scaling)

    def _calculate_sharpe(self):
        ann_ret = self._calculate_mean_ann_return()
        ann_vol = self._calculate_ann_vol()
        return ann_ret / ann_vol

    # ---------------------------
    # weight & exposure analysis
    # ---------------------------

    def _calculate_gross_exposure(self):
        return self.portfolio_weights.abs().sum(axis=1)

    def _calculate_net_exposure(self):
        return self.portfolio_weights.sum(axis=1)

    def _calculate_avg_gross_exposure(self):
        return self._calculate_gross_exposure().mean()

    def _calculate_avg_net_exposure(self):
        return self._calculate_net_exposure().mean()

    def calculate_results_overview(self):
        results_overview = pd.Series(index=['mean_ann_return', 'ann_vol', 'sharpe', 'avg_daily_turnover', 'max_dd',
                                            'max_tuw', 'avg_gross_exposure', 'avg_net_exposure'])
        results_overview.loc['mean_ann_return'] = np.round(self._calculate_mean_ann_return(), 2)
        results_overview.loc['ann_vol'] = np.round(self._calculate_ann_vol(), 2)
        results_overview.loc['sharpe'] = np.round(self._calculate_sharpe(), 2)
        results_overview.loc['avg_daily_turnover'] = np.round(self._calculate_turnover_series().mean(), 2)
        results_overview.loc['max_dd'] = np.round(self._calculate_max_drawdown().values, 2)
        results_overview.loc['max_tuw'] = self._calculate_max_time_under_water().values.astype('timedelta64[D]')
        results_overview.loc['avg_gross_exposure'] = np.round(self._calculate_avg_gross_exposure(), 2)
        results_overview.loc['avg_net_exposure'] = np.round(self._calculate_avg_net_exposure(), 2)
        results_overview.name = self.name

        return results_overview


class Comparis(object):

    def __init__(self, **kwargs):
        self.strategies = DataSet()
        for key, value in kwargs.items():
            self.strategies[key] = value

    def create_results_comparison(self):
        results = pd.DataFrame()
        for name, result in self.strategies.items():
            results = pd.concat([results, result.calculate_results_overview()], axis=1)
        format_dict = {'mean_ann_return': '{:.2%}'}
        # results.style.format(format_dict)
        return results.sort_values(by='sharpe', axis=1, ascending=False)

    def plot_cumulative_returns(self):
        navs = pd.DataFrame()
        for item, strategy in self.strategies.items():
            nav = strategy.net_asset_value.copy()
            nav.name = strategy.name
            navs = pd.concat([navs, nav], axis=1)
        navs.dropna().plot()
