import datetime

import pandas as pd

from classes.Fred import FRED


class YieldCurve(object):

    def __init__(self, ctry, start_date='12/31/2018'):
        self.ctry = ctry
        self.series_id_dict = None
        self.start_date = start_date

    def set_series_id_dict(self):
        if self.ctry == 'US':
            from configs.fred_series_ids import us_yield_curve
            self.series_id_dict = us_yield_curve
        else:
            pass

    def load_yield_curve_data(self):
        end_date = datetime.date.today().strftime('%m/%d/%Y')
        df = pd.DataFrame()
        if self.series_id_dict is None:
            self.set_series_id_dict()
        for key in self.series_id_dict:
            series = FRED(self.series_id_dict[key]).get_fred_time_series(observation_start=self.start_date,
                                                                         observation_end=end_date)
            df[key] = series
            print('data for series {} successfully queried'.format(key))
        return df.dropna()
