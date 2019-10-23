import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch


def calculate_returns(prices, cleaned=True):
    if cleaned:
        returns = np.log(prices).diff(1).ffill(0).dropna(axis=0, how='all')
    else:
        returns = np.log(prices).diff(1)
    return returns


def calculate_simple_covariance(returns):
    cov = returns.cov()
    return cov


def calculate_simple_correlation(returns):
    corr = returns.corr()
    return corr


def calculate_distance_matrix(corr):
    dist = ((1 - corr) / 2.) ** .5  # distance matrix
    return dist


def calculate_linkage_matrix(distance, link_type='single'):
    link = sch.linkage(distance, link_type)
    return link


def calculate_diagonalized_index(link):
    # from link to diagonalization, outputs a list of indexes
    link = link.astype(int)
    sorted_instrument_index = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sorted_instrument_index.max() >= num_items:
        sorted_instrument_index.index = range(0, sorted_instrument_index.shape[0] * 2, 2)  # make space
        df0 = sorted_instrument_index[sorted_instrument_index >= num_items]  # find clusters
        i = df0.index
        j = df0.values - num_items
        sorted_instrument_index[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sorted_instrument_index = sorted_instrument_index.append(df0)  # item 2
        sorted_instrument_index = sorted_instrument_index.sort_index()  # re-sort
        sorted_instrument_index.index = range(sorted_instrument_index.shape[0])
    return sorted_instrument_index.tolist()


# ---------------------------
# inverse vol weighting of a portfolio, and a cluster within (which requires a sorted covmat (sorted by distance)
# ----------------------------

def get_inverse_volatility_weights(covariance):
    ivp = 1. / np.diag(covariance)  # inverse volatility
    ivp /= ivp.sum()  # scale to one
    return ivp


def get_iv_cluster_variance(cov, c_items):
    """
    calculates the variance of an inverse volatility portfolio, where c_items
    identifies which items are to be chosen from larger covariance matrix
    :param cov: covariance as pandas df
    :param c_items: index of columns for cluster, a list
    :return:
    """
    cov_ = cov.loc[c_items, c_items]  # matrix slice
    w_ = get_inverse_volatility_weights(cov_).reshape(-1, 1)  # reshape to column vector
    c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]  # compute variance, get value back
    return c_var
