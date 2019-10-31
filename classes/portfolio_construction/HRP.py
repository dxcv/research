import warnings

import pandas as pd

from risk_metrics.cleaning_routines import calculate_clean_correlation_matrix, calculate_cleaned_cov_mat
from risk_metrics.measures import calculate_linkage_matrix, calculate_simple_covariance, calculate_returns, \
    calculate_diagonalized_index, get_iv_cluster_variance, calculate_distance_matrix, calculate_simple_correlation
from utils.parallel_computing import mp_pandas_obj
from utils.universe_helpers import find_universe_tickers
from utils.nan_handling import find_first_valid_date_where_n_larger_one


warnings.filterwarnings("ignore")


class HRP(object):

    def __init__(self, prices, window_length, **kwargs):
        self.prices = prices.dropna(how='all')  # drop all dates for which I have now valid price
        self.weights = None
        self.constraints = None
        self.dates = prices.index
        self.window_length = window_length
        if 'future_returns' in kwargs:
            self.returns = kwargs['future_returns']
        else:
            self.returns = calculate_returns(self.prices)
        self.hrp_weights = pd.DataFrame(index=self.returns.index, columns=self.returns.columns)
        if 'cleaning' in kwargs:
            self.cleaning = True
        else:
            self.cleaning = False

    @staticmethod
    def _calculate_recursive_bisection(covariance, sorted_index):
        weights = pd.Series(1, index=sorted_index)
        cluster_items = [sorted_index]  # initialize all items in one cluster
        while len(cluster_items) > 0:
            # bi-sect list of cluster items in 2
            cluster_items = [i[int(j): int(k)] for i in cluster_items for j, k in
                             ((0, len(i) / 2), (len(i) / 2, len(i))) if
                             len(i) > 1]
            for i in range(0, len(cluster_items), 2):
                # parse in pairs
                cluster_items_0 = cluster_items[i]
                cluster_items_1 = cluster_items[i + 1]
                cluster_var_0 = get_iv_cluster_variance(covariance, cluster_items_0)
                cluster_var_1 = get_iv_cluster_variance(covariance, cluster_items_1)
                # weight for cluster 1
                alpha = 1 - cluster_var_0 / (cluster_var_0 + cluster_var_1)
                # weight of asset: weight in cluster * weight in next cluster
                weights[cluster_items_0] *= alpha
                weights[cluster_items_1] *= 1 - alpha
        return weights

    def _find_first_valid_optimization_date(self):
        """ first valid opt date is the first opt date for which I have more than one instrument """
        returns = self.returns
        # first date for which I have more than one asset price
        start_date_optimization = find_first_valid_date_where_n_larger_one(returns)
        return start_date_optimization

    def _calculate_hrp(self, covariance, correlation):
        """
        takes a covmat, and a corrmat, calculates distance, linkage matrix, diagonalize by index
        create clusters, inverse-vol weight within cluster
        :param covariance:
        :param correlation:
        :return hrp:  returns hrp allocation for a given return mat
        """
        # takes a covmat, and a corrmat, calculates distance, linkage matrix, diagonalize by index
        # create clusters, inverse-vol weight within cluster
        corr, cov = pd.DataFrame(correlation), pd.DataFrame(covariance)
        distance = calculate_distance_matrix(corr)
        link = calculate_linkage_matrix(distance)
        diagonalized_index = calculate_diagonalized_index(link)
        sorted_correlation_mat_index = corr.index[diagonalized_index].tolist()  # sorted asset names
        hrp = self._calculate_recursive_bisection(cov, sorted_correlation_mat_index)
        return hrp.sort_index()

    def _calculate_hrp_allocation_over_time(self, returns, molecule, events, is_in_universe):
        """
        :param returns:
        :param molecule:
        :param events: pd.Series where index_values are start dates and series values are end dates
        :param is_in_universe: boolean df where for each date is True if date > first valid date
        :return:
        """
        events_ = events.loc[molecule]
        # cut returns matrix to molecule dates
        returns_molecule = returns.loc[events_.index[0]:events_.iloc[-1]]
        # prepare allocation slice
        allocation = pd.DataFrame(columns=returns.columns, index=events_.values)

        # loop through dates within slice, calculate HRP allocation
        for date in events_:
            start_date = events_[events_ == date].index[0]
            return_window = returns_molecule.loc[start_date:date].copy()
            # cut out markets for which current date is before the first valid date
            valid_markets = is_in_universe.columns[is_in_universe.loc[start_date]]
            return_window = return_window[valid_markets].copy()

            if self.cleaning:
                correlation = calculate_clean_correlation_matrix(return_window)
                covariance = calculate_cleaned_cov_mat(return_window)
            else:
                correlation = calculate_simple_correlation(return_window)
                covariance = calculate_simple_covariance(return_window)
            allocation_for_window = self._calculate_hrp(covariance, correlation)
            allocation.loc[date, valid_markets] = allocation_for_window
        return allocation

    def hrp_calculation_through_time_in_parallel(self):
        # parallelize hrp allocation through time
        returns = self.returns.loc[self._find_first_valid_optimization_date():]
        univ_boolean = find_universe_tickers(returns)
        events = pd.Series(data=returns.index[self.window_length:], index=returns.index[:-self.window_length])
        df0 = mp_pandas_obj(func=self._calculate_hrp_allocation_over_time, pd_obj=('molecule', events.index),
                            numThreads=8,
                            returns=returns, events=events, is_in_universe=univ_boolean)
        self.hrp_weights = df0
        return df0

#
# def get_equal_weights(cov):
#     w = np.ones(cov.shape[0])/cov.shape[0]
#     return w
#
#
# def plot_corr_matrix(corr, labels=None, path=None, ax=None):
#     if ax is None:
#         if labels is None:
#             labels = []
#         mpl.pcolor(corr)
#         mpl.colorbar()
#         mpl.yticks(np.arange(.5, corr.shape[0] + .5), labels)
#         mpl.xticks(np.arange(.5, corr.shape[0] + .5), labels)
#         if path is not None:
#             mpl.savefig(path)
#             mpl.clf(); mpl.close()
#     else:
#         if labels is None:
#             labels = []
#         ax.pcolor(corr)
#         mpl.colorbar()
#         mpl.yticks(np.arange(.5, corr.shape[0] + .5), labels, ax=ax)
#         mpl.xticks(np.arange(.5, corr.shape[0] + .5), labels, ax=ax)
#         if path is not None:
#             ax.savefig(path)
#             ax.clf(); ax.close()
#     return
#
#
