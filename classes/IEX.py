import os
from datetime import datetime as dt

import pandas as pd
import requests
from trading_calendars import get_calendar
import pathlib

# todo: pass instrument dataset object which then includes the relevant meta data
# todo: read data from disk, check most current data point, download newest data (self.update_data)


class ConnectionIEX(object):
    """
    class preparing generic API url request
    """

    def __init__(self):
        self.host = 'https://cloud.iexapis.com/'
        self.token = os.environ.get("IEX_TOKEN")
        self.token_string = '&token=' + self.token
        self.version = 'stable/'
        self.category = 'stock/'
        self.query_root = self.host + self.version + self.category
        self.root_folder = 'C:/Users/28ide/Data/'


class HistoricalIntradayIEX(ConnectionIEX):
    """
    class to store (and save to disk) intraday minute bar data from IEX cloud
    """

    def __init__(self, meta_data_market, date):
        """
        :param meta_data_market: market meta data pd.Series (including Market:...)
        :param date: date in YYYYMMDD format
        """
        ConnectionIEX.__init__(self)
        self.symbol = meta_data_market.loc['Symbol']
        self.date_string = date
        self.date = dt.strptime(self.date_string, '%Y%m%d').date()
        self._intraday_data = None
        self.meta_data = meta_data_market

    def create_intraday_query_string(self):
        # todo: add option to add a list of tickers rather than just one symbol, they need to be comma-separated
        filter_values = "?filter=close,date,minute"
        url_string = self.query_root + self.symbol + '/chart/date/' + self.date_string + \
            filter_values + self.token_string
        return url_string

    @property
    def intraday_data(self):
        if self._intraday_data is None:
            self.request_intraday_data()
        return self._intraday_data

    def check_for_valid_date(self):
        """ checks whether the queried date is a holiday
        :return is_trading_day: boolean with 1 if is a trading day else 0
        """
        holidays = get_calendar(self.meta_data['TradingCalendar']).day.calendar.holidays
        is_trading_day = True
        if self.date in holidays:
            is_trading_day = False
            print('Supplied date {} is a holiday. No intraday data is available !!!'.format(self.date))
        return is_trading_day

    def request_intraday_data(self):
        """
        :return df: data frame with intraday data
        """

        if self.check_for_valid_date() is False:
            # todo: potentially find last valid trading date
            raise ValueError
        url = self.create_intraday_query_string()
        request = requests.get(url).json()
        # store as a dataframe with index DateTime
        df = pd.DataFrame(request)
        df['date'] = pd.DatetimeIndex(df['date'] + ' ' + df['minute'])
        df = df.drop('minute', axis=1).set_index('date')
        # store value as object attribute
        self._intraday_data = df
        return df


class HistoricalEodIEX(ConnectionIEX):

    def __init__(self, symbol, date, meta_data):
        ConnectionIEX.__init__(self)
        self.date_string = date
        self.date = dt.strptime(self.date_string, '%Y%m%d').date()
        self.symbol = symbol
        self.meta_data = meta_data
        self._time_series_max = None

    @property
    def time_series_max(self):
        """ get maximum time series available from IEX """
        if self._time_series_max is None:
            self.request_time_series_max()
        return self._time_series_max

    def create_max_string(self, attribute):
        """ creates string for maximum query """
        filter_values = "?filter=" + attribute + ",date"
        url_string = self.query_root + self.symbol + '/chart/max/' + filter_values + self.token_string
        return url_string

    def request_time_series_max(self, attribute='close'):
        """
        gets maximum time series from IEX for an attribute, e.g. close
        :return df: time series df
        """
        # todo: check whether to use uClose
        # create query string
        url = self.create_max_string(attribute)
        # request data
        request = requests.get(url).json()
        # store as a dataframe with index DateTime
        df = pd.DataFrame(request)
        df['date'] = pd.DatetimeIndex(df['date'])
        df = df.set_index('date')
        # store value as object attribute
        self._time_series_max = df
        return df

    def write_data_to_disk(self):
        data = self.time_series_max
        ac_folder = self.meta_data.SecType
        file_name = self.symbol
        data_folder = self.root_folder
        # create folder if necessary
        folder = pathlib.Path(data_folder + ac_folder + '/')
        if folder.exists() is False:
            folder.mkdir(parents=True, exist_ok=True)
        file = data_folder + ac_folder + '/' + file_name + '.parquet.gzip'
        data.to_parquet(file, compression='gzip')
        print('*** data written to disk ***')

    def update_data(self):
        return NotImplementedError

# class SampleObject:
#
#     def __init__(self):
#         # ...
#         self._total = None
#
#     @property
#     def total(self):
#         """Compute or return the _total attribute."""
#         if self._total is None:
#             self.compute_total()
#
#         return self._total
