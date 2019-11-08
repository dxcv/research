import numpy as np
import pandas as pd

from utils.parallel_computing import mp_pandas_obj


def calculate_concurrent_events_per_date(price_index, touch_dates, molecule):
    """
    compute number of events per bar: for a date how in how many ranges (tb-intervalls) is a given price
    If I sample a price, that info will be in N different labels (that's the info I want to extract)
    :param price_index: pd.DatetimeIndex
    :param touch_dates: labels endtime, 'barrier_date' (from get_barrier_touch_dates), should be a series!
    :param molecule: pd.DatetimeIndex with event start dates (only really used for parallelization)
    :return:
    """
    if type(touch_dates) == pd.DataFrame:
        touch_dates = touch_dates['barrier_date'].copy()
    touch_dates = touch_dates.fillna(price_index[-1])  # set end date for those labels that are not closed
    touch_dates = touch_dates[touch_dates >= molecule[0]]  # events that end at or after molecule[0] date
    touch_dates = touch_dates.loc[:touch_dates[molecule].max()]  # events that start before final date
    # count concurrent events in a bar/date
    iloc = price_index.searchsorted(np.array([touch_dates.index[0], touch_dates.max()]))
    count = pd.Series(0, index=price_index[iloc[0]:iloc[1] + 1])
    for start, end in touch_dates.iteritems():
        count.loc[start:end] += 1
    return count.loc[molecule[0]:touch_dates[molecule].max()]


def calculate_average_uniqueness_per_date(touch_dates, concurrent_events, molecule):
    """
    calculates the average historical uniqueness of a label (1 over concurrency of prices in range) averaged over time
    the more a price is in different labels, i.e. the higher the concurrency, the lower the uniqueness
    :param touch_dates: from barrier calculation, should be a series
    :param concurrent_events: calculated from above
    :param molecule: pd.DatetimeIndex
    :return: "sample" weights, to add for drawing sample in ML optimization
    """

    if type(touch_dates) == pd.DataFrame:
        touch_dates = touch_dates['barrier_date'].copy()  # convert to series
    wghts = pd.Series(index=molecule)
    for start, end in touch_dates.loc[wghts.index].iteritems():
        wghts.loc[start] = (1 / concurrent_events.loc[start:end]).mean()  # average uniqueness of label
    return wghts


def calculate_sampling_weights_from_avg_label_uniqueness(barrier_dates, prices):
    """ takes barrier weights from tb-labelling, add weight info as column to bin (from labelleling) for further
    processing, sample weights to pick when choosing (y,x) pair for ml algo """
    cnc = mp_pandas_obj(calculate_concurrent_events_per_date,
                        ('molecule', barrier_dates.index),
                        numThreads=8, price_index=prices.index, touch_dates=barrier_dates['barrier_date'])
    concurrent_events = cnc.loc[~cnc.index.duplicated(keep='last')]
    concurrent_events = concurrent_events.reindex(prices.index).fillna(0)
    weights = mp_pandas_obj(calculate_average_uniqueness_per_date, ('molecule', barrier_dates.index), numThreads=8,
                            touch_dates=barrier_dates['barrier_date'], concurrent_events=concurrent_events)
    return weights


# -------------------------
# sequential bootstrap
# -------------------------


def calculate_indicator_matrix(prices_dates, barrier_dates):
    """
    :param prices_dates: pd.DatetimeIndex
    :param barrier_dates: pd.Series with start, end in index, col respectively (barrier_dates from tb-calculation
    :return indicator_matrix:
    """
    if type(barrier_dates) == pd.DataFrame:
        barrier_dates = barrier_dates['barrier_date'].copy()
    indicator_matrix = pd.DataFrame(0, index=prices_dates, columns=range(barrier_dates.shape[0]))
    for label, (start, end) in enumerate(barrier_dates.iteritems()):
        indicator_matrix.loc[start:end, label] = 1
        # print('calculated indicator entries for date {}'.format(start))
    return indicator_matrix


def calculate_average_uniqueness_of_samples(indicator_matrix):
    """
    :param indicator_matrix: T x (nbr of events) one if price t in event t'
    :return: avg_u: average uniequeness
    """
    conc = indicator_matrix.sum(axis=1)  # any t price in N labels
    uniqueness = indicator_matrix.div(conc, axis=0)  # uniqueness
    avg_uniqueness = uniqueness[uniqueness > 0].mean()
    return avg_uniqueness


def run_sequential_bootstrap(indicator_matrix, sample_length=None):
    """
    performs a sequential bootstrap on a given indicator matrix for a sample of length n
    :param indicator_matrix: indicator matrix from above
    :param sample_length: how many samples to be drawn
    :return: index of samples drawn [ilocs]
    """
    if sample_length is None:
        sample_length = indicator_matrix.shape[1]  # nbr of events
    phi = []
    while len(phi) < sample_length:
        avg_uniqueness = pd.Series()
        for event in indicator_matrix:
            indicator_matrix_ = indicator_matrix[phi + [event]]
            avg_uniqueness.loc[event] = calculate_average_uniqueness_of_samples(indicator_matrix_).iloc[-1]
        probability = avg_uniqueness / avg_uniqueness.sum()  # draw probability
        phi += [np.random.choice(indicator_matrix.columns, p=probability)]
        print('Sample No. {} drawn'.format(phi))
    return phi


# time decay
def calculate_time_decay(weights, maximal_decay):
    """
    piece-wise linearly decay of weights the further into the past we go.
    :param weights: takes e.g. time-avg uniqueness weights
    :param maximal_decay: scaling of final weight: e.g. if 1 there is no time scaling, if 0 then first sample has 0 wgt
    :return: time_decay sampling weights: multiply by avg_uniqueness wgts for
    """
    decayed_wgts = weights.sort_index().cumsum()
    if maximal_decay >= 0:
        slope = (1 - maximal_decay) / decayed_wgts.iloc[-1]
    else:
        slope = 1 / ((maximal_decay + 1) * decayed_wgts.iloc[-1])
    constant = 1 - slope * decayed_wgts.iloc[-1]
    decayed_wgts = constant + slope * decayed_wgts
    decayed_wgts[decayed_wgts < 0] = 0
    print("slope value: {}; constant: {}; ".format(slope, constant))
    return decayed_wgts
