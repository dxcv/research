import numpy as np
import pandas as pd

from utils.parallel_computing import mp_pandas_obj


def get_barrier_touch_dates(prices, dates, pt_sl, trgt, min_ret, num_threads, vertical_barrier=False, meta_label=False):
    """
    calculates barrier touch dates for any date in dates: either by pt_sl or by vertical barrier
    :param prices: price series pd.Series
    :param dates: pd.DatetimeIndex of dates for which barrier touches should be calculated
    :param pt_sl: float setting the width of the horizontal barrier (0 means no horizontal barrier)
    :param trgt: pandas series of targets (for horizontal barrier), expressed in terms of absolute returns
    :param min_ret: minimum target return required for running a tripple barrier search
    :param num_threads:
    :param vertical_barrier: pd.Series of timestamps of vertical barriers
    :param if metal_label is false : we don't know the side of the trade
    :return: touch_params: pd.Df with columns barrier_date, trgt
    """

    trgt = trgt.loc[dates]
    # only use sensible trgts, e.g. if no vol over a period  (abs return e.g. 1 vol) don't use it as a target
    trgt = trgt[trgt > min_ret]
    if vertical_barrier is False:
        vertical_barrier = pd.Series(pd.NaT, index=dates)
    # form touches, apply stop loss on vertical barrier
    if meta_label is False:
        side_ = pd.Series(1, index=trgt.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = meta_label.loc[trgt.index]
        pt_sl_ = pt_sl[:2]
    touch_params = pd.concat({'barrier_date': vertical_barrier, 'trgt': trgt, 'side': side_}, axis=1).dropna(
        subset=['trgt'])
    df = mp_pandas_obj(func=run_tripple_barrier_date_search, pd_obj=('molecule', touch_params.index),
                       numThreads=num_threads,
                       prices=prices, barrier_info=touch_params, pt_sl=pt_sl_)
    touch_params['barrier_date'] = df.dropna(how='all').min(
        axis=1)  # get earliest touch date out of, vb, sl, pt touches
    if meta_label is False:
        touch_params = touch_params.drop('side', axis=1)
    touch_params = touch_params.drop('trgt', axis=1)
    return touch_params


def run_tripple_barrier_date_search(prices, barrier_info, pt_sl, molecule):
    """
    master function returning barrier date (either vb, sl, pt) per start date
    :param prices: pd.Series of prices
    :param barrier_info: df with columns 'barrier_date and trgt',
    :param pt_sl: list of 2 non-neg scalars with which to multiply target
    :param molecule: list of event dates that will be processed in a single thread
    :return: out: df with vb, pt, sl dates per date (pd.NaT if a barrier is not touched)
    """
    events_ = barrier_info.loc[molecule]
    out = events_[['barrier_date']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['trgt']
    else:
        pt = pd.Series(index=barrier_info.index)
    if pt_sl[1] > 0:
        sl = - pt_sl[1] * events_['trgt']
    else:
        sl = pd.Series(index=barrier_info.index)

    for loc, vb in events_['barrier_date'].fillna(prices.index[-1]).iteritems():
        df = prices[loc:vb]
        df = (df / prices[loc] - 1) * events_.at[loc, 'side']  # changes signs if necessary, path return
        out.loc[loc, 'sl'] = df[df < sl[loc]].index.min()  # earliest stop loss
        out.loc[loc, 'pt'] = df[df > pt[loc]].index.min()  # earliest profit taking
    return out


def get_daily_vol(prices, span=100):
    # todo: I should use the util function
    df = prices.index.searchsorted(prices.index - pd.Timedelta(days=1))
    df = df[df > 0]
    df = pd.Series(prices.index[df - 1], index=prices.index[prices.shape[0] - df.shape[0]:])
    df = prices.loc[df.index] / prices.loc[df.values].values - 1
    df = df.ewm(span=span).std()
    return df


def set_vertical_barrier(dates, days):
    """
    :param dates: dates index
    :param days: num of days to be shifted
    :return: vb: pd.Series with start_date, barrier_date
    """
    vb = dates.searchsorted(dates + pd.Timedelta(days=days))
    vb = vb[vb < len(dates)]
    vb = pd.Series(dates[vb], index=dates[:vb.shape[0]])
    return vb


def get_bins_sign(barrier_dates, prices):
    """
    calculate bins: +1 if pos tb-return, -1 in neg tb-return. I need to feed in the barrier dates.
    :param barrier_dates: dates to calculate return (calculated from get_barrier_touch_dates), if contains side then meta
    :param prices:
    :return: -1, 1 if usual, 0, 1 if meta-labelling
    """
    events_ = barrier_dates.dropna(subset=['barrier_date'])
    px = events_.index.union(events_['barrier_date'].values).drop_duplicates()
    px = prices.reindex(px, method='bfill')
    # create output
    df = pd.DataFrame(index=events_.index)
    df['return'] = px.loc[events_['barrier_date'].values].values / px.loc[events_.index] - 1
    if 'side' in events_:
        df['return'] *= events_['side']
    df['bin'] = np.sign(df['return'])
    if 'side' in events_:
        df.loc[df['return'] < 0, 'bin'] = 0  # meta-labelling adjustment
    return df


def get_labels_from_tripple_barrier_returns(prices, span=100, pt_sl=[1, 1], min_ret=0, num_threads=8, vb_days=10,
                                            meta_label=False):
    """
    takes a price series (and the relevant parametrization) and classifies a
    prices series according to the signed tripple barrier return"
    "label" looks into the future at date t (index value) df.return shows subsequent tb-return, label=shows label
    dates are propperly aligned, now shifting need at further analysis
    :param prices:
    :param span: to calculate target_vol unit
    :param pt_sl: list: 1 for horizontal barriers at trgt_vol
    :param min_ret: don't use hbs with too low a barrier
    :param num_threads: for parallelization
    :param vb_days: how many days for vertical barrier
    :param meta_label: if False no side info passed, else I pass a Series indicating side of trade
    :return tp_bins: df with columns return and bin
    """
    trgt_hb_in_unit_vol = get_daily_vol(prices, span)
    vertical_barrier = set_vertical_barrier(dates=prices.index, days=vb_days)
    barrier_dates = get_barrier_touch_dates(prices=prices, dates=prices.index, pt_sl=pt_sl, trgt=trgt_hb_in_unit_vol,
                                            min_ret=min_ret, num_threads=num_threads,
                                            vertical_barrier=vertical_barrier, meta_label=meta_label)
    tp_bins = get_bins_sign(barrier_dates, prices)
    return tp_bins, barrier_dates
