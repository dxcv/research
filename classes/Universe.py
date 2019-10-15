import pandas as pd
from configs.data_config import set_data_config
from classes.Instrument import Instrument
from utils.dataset import DataSet
# loads data from hard - disk
# loads current data from api


class Universe(object):
    """ Universe class which holds market meta data which is read from JSON """

    def __init__(self, universe, frequency):
        """
        :param universe: the group of symbols for which meta data is loaded, see configs/universe.py
        """
        self.data_config = set_data_config(universe)
        self._universe_meta_data = None
        self.frequency = frequency
        self.symbols = self.data_config.symbols_list
        self.raw_data = DataSet()

    @property
    def universe_meta_data(self):
        if self._universe_meta_data is None:
            self.read_meta_data_for_universe()
        return self._universe_meta_data

    def read_meta_data_for_universe(self):
        """ reads json universe data
        :return: df with symbols as index, attributes as columns
        """

        df = pd.read_json(self.data_config.meta_data_root).transpose().loc[self.data_config.symbols_list]
        self._universe_meta_data = df
        return df

    @staticmethod
    def read_instrument_data(symbol, data, target_freq):
        instrument = Instrument(symbol, data, target_freq)
        return instrument

    def read_universe_data(self, raw_data_set):
        for symbol in self.universe_meta_data.index:
            value = self.read_instrument_data(symbol, raw_data_set[symbol], self.frequency)
            self.raw_data[symbol] = value

    def get_cross_sectional_view(self, attribute):
        df = pd.DataFrame()
        for symbol in self.symbols:
            df_tmp = getattr(self.raw_data[symbol], attribute)
            df_tmp.name = symbol
            df = pd.concat([df, df_tmp], axis=1)
        return df
