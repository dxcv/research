import numpy as np

from utils.risk_metrics import calculate_exp_moving_vol
from utils.resampling_and_risk_metrics import calculate_samples_per_day



# todo: samples_per_day will be inferred from instrument.samples_per_day which calculates its value from
#  utils.resampling.calculate_samples_per_day()


def calculate_ts_momentum(ts, leverage=2):
    """
    :param: ts as a pd.Series
    from Kelly et al. Factor Momentum everywhere
    :return:
    """
    vol = calculate_exp_moving_vol(ts)
    # todo: formula


def oscillator_norm(fast, slow, samples_per_day):
    # calculates oscillator norm to adjust signal for volatility of filter/CMA
    # fast is the short window, slow the long window
    x = 1 * ((fast - 1) / fast)
    y = 1 * ((slow - 1) / slow)
    increment = 1 / samples_per_day

    assert fast < slow, "Fast lookback %d should be shorter than slow lookback %d" % (fast, slow)

    norm = np.sqrt(increment / (1 - x ** (2 * increment)) +
                   increment / (1 - y ** (2 * increment)) -
                   2 * increment / (1 - (x * y) ** increment))

    return norm


def exp_ma(series, window):
    # calculates an exponential moving average over specified window
    # todo: take away samples per day; infer frequency and then let run (in oscillator norm, exp_ma, exp_mvol,...)
    samples_per_day = calculate_samples_per_day(series.index.freq)
    window = samples_per_day * window
    span = 2 * window - 1
    min_periods = 2 * window

    assert len(series) > min_periods, "Not enough data to calculate exponential moving average, " \
                                      "min period %d > series length" % min_periods

    return (series.ewm(span=span, min_periods=min_periods, adjust=False)).mean()


def exp_ma_crossover(series, fast, slow, window, vol_floor=0):
    # calculates the ma crossover signal, adjusted for the vol of the filter/signal and vol of the instrument

    assert fast < slow, "Fast lookback %d should be shorter than slow lookback %d" % (fast, slow)

    samples_per_day = calculate_samples_per_day(series.index.freq)
    fast_ma = exp_ma(series, fast)
    slow_ma = exp_ma(series, slow)
    ma_crossover = fast_ma - slow_ma
    norm = oscillator_norm(fast, slow, samples_per_day=samples_per_day)
    daily_vol = calculate_exp_moving_vol(series.diff(periods=1), window)  # USD vol
    daily_vol = np.maximum(daily_vol.fillna(0), vol_floor)

    return ma_crossover / (norm * daily_vol)
