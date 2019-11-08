import pandas as pd
from utils.dataset import DataSet


def symmetric_cusum_filter(data, threshold):

    # set up containers for date_index, pos and negative sums
    dates, pos_sum, neg_sum = [], 0, 0
    diff = data.diff()
    pos_series = pd.Series(index=diff.index)
    neg_series = pd.Series(index=diff.index)

    # calculate cumulative sums and reset if > or below threshold
    for date in diff.index[1:]:
        pos_sum, neg_sum = max(0, pos_sum + diff.loc[date]), min(0, neg_sum + diff.loc[date])
        pos_series.loc[date] = pos_sum
        neg_series.loc[date] = neg_sum
        if neg_sum < - threshold:
            neg_sum = 0
            dates.append(date)
        elif pos_sum > threshold:
            pos_sum = 0
            dates.append(date)

    cumulative_series = pd.concat([pos_series.cumsum(), neg_series.cumsum()], axis=1)
    cumulative_series.columns = ['pos_series', 'neg_series']

    cusum_filter = DataSet()
    cusum_filter.dates = pd.DatetimeIndex(dates)
    cusum_filter.threshold = threshold
    cusum_filter.cumulative_series = cumulative_series

    return cusum_filter

