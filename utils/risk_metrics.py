import numpy as np

from utils.resampling_and_risk_metrics import calculate_samples_per_day

def calculate_exp_moving_vol(series, window, min_periods=0):
    # calculates exponential moving vol

    # infer frequency
    samples_per_day = calculate_samples_per_day(series.index.freq)
    window = samples_per_day * window
    span = 2 * window - 1
    min_periods = np.maximum(window, min_periods)

    assert len(series) > min_periods, "Not enough data to calculate exponential vol, " \
                                      "min period %d > series length" % min_periods

    exp_mvol = (series.ewm(span=span, min_periods=min_periods, adjust=False)).std(bias=False)

    return exp_mvol


def calculate_exp_vol(series, min_periods=0):
    # calculates exponential moving vol

    # infer frequency
    # samples_per_day = calculate_samples_per_day(series.index)
    window = series.shape[0]
    span = 2 * window - 1

    assert len(series) > min_periods, "Not enough data to calculate exponential vol, " \
                                      "min period %d > series length" % min_periods

    exp_vol = (series.ewm(span=span, adjust=False)).std(bias=False)[-1]

    return exp_vol


# def calculate_samples_per_day(index):
#     """
#     :param index: time series index
#     :return: scaling_factor: scalar
#     """
#     frequency = index.freq
#     if frequency == '1H':
#         scaling_factor = 24
#     elif frequency == '1D':
#         scaling_factor = 1
#     elif frequency == 'S':
#         scaling_factor = 24 * 60 * 60
#     elif frequency == '1min':
#         scaling_factor = 24 * 60
#     else:
#         scaling_factor = None
#         print("Inaccurate frequency provided")
#     return scaling_factor
