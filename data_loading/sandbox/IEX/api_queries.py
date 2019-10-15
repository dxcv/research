from iexfinance.stocks import get_historical_data
from iexfinance.stocks import get_historical_intraday
from datetime import datetime as dt


def get_time_series_eod(symbol, start_date, end_date=dt.now(), close_only=False):
    """
    queries EOD stock data
    :param symbol: IEX symbol, type: string from from iexfinance.refdata import get_iex_symbols
    :param start_date: from, type: datetime.datetime
    :param end_date: to, type: datetime.datetime
    :param close_only: True if only close data to be queries
    :return df: pd.DataFrame
    """

    df = get_historical_data(symbol, start_date, end_date, close_only=close_only)
    return df


def get_time_series_intra_day(symbol, date=dt.now()):
    """
    :param symbol:
    :param date: type datetime.datetime
    :return: df: pd.DataFrame
    """

    df = get_historical_intraday(symbol, date=date)
    return df


def save_time_series_to_disk(df, file_name, path):
    """
    stores time series to disk
    :param df: time series to be written to disk
    :param file_name: file name string
    :param path: path string
    :return:
    """
    #todo: should load path from config file
    df.to_parquet(fname=path + file_name + '.parq')
    return


# when resampling intraday: add label='right