import numpy as np
import pandas as pd
from scipy import optimize
from risk_metrics.cleaning_routines import calculate_cleaned_cov_mat
from utils.universe_helpers import find_universe_tickers


class RiskParity:

    def __init__(self, time_series, window=30, upper_bound=None, lower_bound=None, **kwargs):
        self.prices = time_series
        self.allocation = None
        self.risk_budgets = None
        self.window_length = window
        self.upper_bound = upper_bound  # lower and upper bound
        self.lower_bound = lower_bound
        if 'futures_returns' in kwargs:
            self.returns = kwargs['futures_returns']
        else:
            self.returns = self.calculate_returns(self.prices)

    @staticmethod
    def calculate_returns(prices, how='log'):
        if how == 'log':
            returns = np.log(prices).diff(1)
        else:
            assert how == 'disc'
            returns = prices/prices.shift(1) - 1
        return returns

    @staticmethod
    def calculate_covariance(returns):
        clean_cov = calculate_cleaned_cov_mat(returns)

        return clean_cov

    def _calculate_risk_allocation(self, date, active_markets):

        # get window length and raw prices
        window = self.window_length

        # prepare price window, cut for active markets and dates in date window
        current_date_loc = self.returns.index.get_loc(date)
        first_date = self.returns.index[current_date_loc - window] # window + 1 observiations


        # calculate returns from window
        returns = self.returns.loc[first_date:date]
        n_assets = returns.shape[1]

        # calculate covariance
        cov_mat = self.calculate_covariance(returns)  # pca_cleaning

        # Ignore correlations < 0
        cov_mat[cov_mat < 0] = 0
        budget = np.array(np.ones((n_assets, 1)) * 1 / n_assets)  # numpy matrix nAssets x 1

        wgts = np.ones((n_assets, 1)) * 1 / n_assets
        cons = ({'type': 'eq', 'fun': lambda wgts: sum(wgts) - 1})

        def rb_function(wgts, cov_mat, budget, n_assets):
            cov_mat = np.array(cov_mat)
            wgts = wgts.reshape(n_assets, 1)
            RC = cov_mat.dot(wgts)  # vector of risk contributions: np.array nAssets x 1
            TR = wgts.T.dot(cov_mat).dot(wgts)  # 1x1 np.array

            value_vector = np.power(np.multiply(wgts, RC) / TR - budget, 2)
            function_value = np.sum(value_vector)

            return function_value

        #todo: different bounds input... check optimize bounds
        res = optimize.minimize(rb_function, wgts, args=(cov_mat, budget, n_assets), method='SLSQP',
                                    bounds=optimize.Bounds(self.lower_bound, self.upper_bound),
                                    constraints=cons, options={'disp': False, 'maxiter': 1000, 'ftol': 1e-6})

        result = res.x

        # Funky results with huge weights occur a couple of times throughout the history
        result[result > 1.0] = 0.0
        result[result < 0.0] = 0.0

        # Set weight for markets where all returns are nan to 0. cc all zero

        return result

    def calculate_rolling_erc_allocation(self):

        dates = self.returns.index[self.window_length + 1:] # first n returns from first n+1 prices


        # erc weights dataframe
        erc_weights = pd.DataFrame(index=dates, columns=self.returns.columns, data=np.nan)
        is_in_universe = find_universe_tickers(self.returns)

        for date in dates:
            valid_markets =  is_in_universe.columns[is_in_universe.loc[date]]
            result = self._calculate_risk_allocation(date, valid_markets)
            erc_weights.loc[date][valid_markets] = result
            if date.is_year_end:
                print("Optimisation ERC for date {} done ".format(date))
            erc_weights.fillna(0, inplace=True)

        self.allocation = erc_weights

        return erc_weights