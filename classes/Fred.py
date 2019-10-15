import fredapi
import pandas as pd


class FRED(object):

    def __init__(self, series_id):
        self.api_key = '5f4c7f58c788ae1f328db20670ae71a2'
        self.series_id = series_id

    def create_connection(self):
        connection = fredapi.Fred(api_key=self.api_key)
        return connection

    def get_fred_time_series(self, observation_start, observation_end):
        # queries fred db, needs series_id, start and end date
        time_series_connection = self.create_connection()
        data_series = time_series_connection.get_series(self.series_id, observation_start, observation_end)
        return data_series


def query_yield_curve_df(series_id_dict, start_date, end_date):
    df = pd.DataFrame()
    for key in series_id_dict:
        series = FRED(series_id_dict[key]).get_fred_time_series(observation_start=start_date, observation_end=end_date)
        df[key] = series
        print('data for series {} successfully queried'.format(key))
    return df.dropna()
