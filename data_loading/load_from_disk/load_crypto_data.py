import pandas as pd


from classes.Universe import Universe
from utils.dataset import DataSet


def load_crypto_data_from_disk(universe_symbols, frequency):
    universe = Universe(universe_symbols, frequency)
    universe_data = DataSet()

    for crypto in universe_symbols:
        data_root = universe.universe_meta_data.loc[crypto]['DataRoot']
        if frequency != '1H':
            data_df = pd.read_csv(data_root + crypto + '_d.csv', skiprows=1, index_col='Date', parse_dates=True).sort_index()
        else:
            data_df = pd.read_csv(data_root + crypto + '_1h.csv', skiprows=1, index_col='Date',
                                  parse_dates=True).sort_index()
            data_df.index = pd.to_datetime(data_df.index, format='%Y-%m-%d %I-%p')
        # todo: proper dynamic frequency
        data_df = data_df.resample(frequency).last()
        for date in data_df.index:
            if data_df.loc[date].isna().sum() == data_df.shape[1]:
                print("Warning: After resampling, no data available for " + crypto + " for date " + str(date))
        universe_data[crypto] = data_df

    universe.read_universe_data(universe_data)

    return universe

# close = universe.get_cross_sectional_view('Close')
# btcusd = universe.raw_data.BTCUSD.get_df_view(['Open', 'High', 'Low', 'Close', 'VolumeUSD'])