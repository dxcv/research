from data_loading.load_from_disk.load_data import load_data_from_disk


def update_data(symbols, frequency, attribute):
    """
    :param symbols: list of symbols, ['NFLX']
    :param frequency: query frequency, e.g. 'B'
    :param attribute: which attribute to use to determine date, e.g. 'close'
    :return:
    """
    disk_data = load_data_from_disk(symbols, frequency)
    most_recent_date = disk_data.get_cross_sectional_view(attribute).index[-1]
    # todo need to add option to determine start query date in classes.IEX request_time series snapshot;
    #  from most recent date until today, add version to time_series_max
    #  inelegant solution: check if date >3m further back than today, create 3_month query, 1_month_query, ...
    return most_recent_date
