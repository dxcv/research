from configs.universe_spec import universe_tech
from classes.Universe import Universe
from classes.IEX import HistoricalIntradayIEX, HistoricalEodIEX
from utils.dataset import DataSet
from data_loading.load_from_disk.load_equities_data import load_equities_data_from_disk


# get meta data
universe = Universe(universe_tech)
univ_meta_data = universe.read_meta_data_for_universe()


# establish IEX connection and read intraday data
dataset = DataSet()

for symbol in univ_meta_data.index:
    # fb_id = HistoricalIntradayIEX(meta_data_market=meta_market, date='20190621')
    # read max data from IEX, then write to disk
    tmp = HistoricalEodIEX(
        symbol=symbol, meta_data=univ_meta_data.loc[symbol], date='20191029')
    dataset[symbol] = tmp.request_time_series_max('close')
    tmp.write_data_to_disk()

# fb = HistoricalEodIEX('FB', '20190715', meta_market)

eq = load_equities_data_from_disk(universe_tech, 'D').get_cross_sectional_view('close').dropna()
# linear_ls_regression(yx=rets[['GOOG', 'NFLX']].dropna()); rets = np.log(eq).diff()
