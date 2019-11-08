import datetime

from classes.Fred import FRED
from utils.dataset import DataSet


def sahm_unemployment_indicator(end_date=None):
    """
    returns a boolean series where True indicates that according to Claudia Sahm indicator economy will be in recession
    :param end_date: unemployment rate
    :return:dataset with indicator
    """
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m')

    # create Fred object and load data
    indicator = FRED('UNRATE')
    data = indicator.get_fred_time_series('1960-1', observation_end=end_date)

    #
    three_month_avg = data.rolling(window=3).mean()
    twelve_month_low = data.rolling(window=12).min()
    difference = three_month_avg - twelve_month_low
    indicator = data[difference > 0.5].index

    sahm_info = DataSet()
    sahm_info.recession_dates = indicator
    sahm_info.indicator = difference
    sahm_info.is_recession = difference > 0.5

    return sahm_info
