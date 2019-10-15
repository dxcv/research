import pandas as pd


class Instrument(object):
    """ Individual Instrument """

    def __init__(self, symbol, df_data, target_freq):
        self.symbol = symbol
        self.raw_data = df_data
        self.date_index = df_data.index
        self.frequency = target_freq
        for key in df_data:
            # todo: only set attribute for certain data (e.g. Close, Open, High,...)
            setattr(self, key.replace(' ', ''), self.raw_data[key])

    def get_df_view(self, attributes_list):
        df = pd.DataFrame()
        for attribute in attributes_list:
            series = getattr(self, attribute)
            series.name = attribute
            df = pd.concat([df, series], axis=1)
        return df

