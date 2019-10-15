import pandas as pd
import numpy as np


def calculate_samples_per_day(frequency):
    if frequency == 'B' or frequency == 'D':
        samples_per_day = 1
    elif frequency == 'min':
        samples_per_day = 24 * 60
    elif frequency == 'H':
        samples_per_day = 24
    else:
        assert frequency == 'S'
        samples_per_day = 24 * 60 * 60
    return samples_per_day


def calculate_log_returns(price_series):
    one_period_returns = np.log(price_series).diff(1)
    return one_period_returns


def calculate_annualization_factor(frequency):
    ann_factor = 260
    samples_per_day = calculate_samples_per_day(frequency)
    ann_factor *= samples_per_day
    return ann_factor


def calculate_annualized_return(price_series, frequency):
    returns = calculate_log_returns(price_series)
    ann_factor = calculate_annualization_factor(frequency)
    return returns.mean() * ann_factor


def calculate_annualized_std(price_series, frequency):
    returns = calculate_log_returns(price_series)
    ann_factor = np.sqrt(calculate_annualization_factor(frequency))
    return returns.std() * ann_factor


def calculate_drawdown_series(price_series):
    max_value = price_series.expanding().max()
    drawdown = (price_series - max_value)/max_value
    return drawdown


def calculate_max_drawdown(price_series):
    drawdown_series = calculate_drawdown_series(price_series)
    maximum_drawdown = pd.Series(data=drawdown_series.min(), index=[drawdown_series.idxmin()])
    return maximum_drawdown


def calculate_time_under_water(price_series):
    drawdown_series = calculate_drawdown_series(price_series)
    dd_df = pd.DataFrame(drawdown_series)
    dd_df['not_in_dd'] = drawdown_series == 0
    time_under_water = dd_df[dd_df['not_in_dd']].index.to_series().diff()
    return time_under_water


def calculate_max_time_under_water(price_series):
    time_under_water = calculate_time_under_water(price_series)
    max_time_under_water = pd.Series(index=[time_under_water.idxmax()], data=time_under_water.max())
    return max_time_under_water


# ---------------------------------

def calculate_herfindahl(rets):
    """
    calculates the concentration of returns, e.g. pos return concentration, or neg return concentration, or time conc
    based on herfindahl index: if result is 0, uniform returns, if return is 1, fully concentrated returns
    :param rets: series of strategy returns
    :return:
    """
    weight = rets / rets.sum()
    hi = (weight**2).sum()
    hi = (hi - rets.shape[0] ** -1) / (1 - rets.shape[0] ** -1)
    return hi


def calculate_sample_stats(price_series, frequency, name):
    # calculate some sample stats and puts them into a pandas Series
    """
    :param price_series: nav
    :param frequency: either 'H', 'S', 'min', 'D' or 'BD'
    :param name: string with the strategy name / description
    :return:
    """
    ann_ret = calculate_annualized_return(price_series, frequency)
    ann_std = calculate_annualized_std(price_series, frequency)
    sharpe_ratio = ann_ret / ann_std
    max_dd = calculate_max_drawdown(price_series)
    max_tuw = calculate_max_time_under_water(price_series)
    stats_series = pd.Series(index=['ann_ret', 'ann_std', 'SR', 'maxDD', 'TuW'])
    stats_series['ann_ret'] = ann_ret
    stats_series['ann_std'] = ann_std
    stats_series['SR'] = sharpe_ratio
    stats_series['maxDD'] = max_dd.values
    stats_series['TuW'] = max_tuw.values
    stats_series.name = name
    return stats_series

