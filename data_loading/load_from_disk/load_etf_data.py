import pandas as pd


from classes.Universe import Universe
from utils.dataset import DataSet


def load_etf_data_from_disk(universe_symbols, frequency):
    universe = Universe(universe_symbols, frequency)
    universe_data = DataSet()

    for etf in universe_symbols:
        data_root = universe.universe_meta_data.loc[etf]['DataRoot'] + '/'
        file_end = '.parquet.gzip'
        data_df = pd.read_parquet(data_root + etf + file_end)
        # todo: proper dynamic frequency
        data_df = data_df.resample(frequency).last()
        for date in data_df.index:
            if data_df.loc[date].isna().sum() == data_df.shape[1]:
                print("Warning: After resamling, no data available for " + etf + " for date " + str(date))
        universe_data[etf] = data_df

    universe.read_universe_data(universe_data)

    return universe
