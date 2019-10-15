import pandas as pd
from utils.dataset import DataSet


class Crypto(object):

    def __init__(self, df):
        """ initiates crypto object, a DataSet is passed, with close prices, volumes, open, etc. """
        self.raw_data = df
        for key in df:
            setattr(self, key.replace(' ', ''), self.raw_data[key])


# pd.read_csv('C:/Users/28ide/Data/Crypto/Coinbase_BTCUSD_d.csv', skiprows=1, index_col='Date')