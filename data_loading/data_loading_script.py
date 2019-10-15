from configs.universe_spec import universe_tech
from classes.Universe import Universe
from classes.Instrument import Instrument
from classes.IEX import HistoricalIntradayIEX, HistoricalEodIEX


# get config

# get meta data
universe = Universe(universe_tech)
univ_meta_data = universe.read_meta_data_for_universe()
symbol = 'NFLX'
fb = Instrument(universe_tech, symbol, data=None)
meta_market = fb.extract_instrument_meta_data()

# establish IEX connection and read intraday data
fb_id = HistoricalIntradayIEX(meta_data_market=meta_market, date='20190621')
fb_eod = HistoricalEodIEX(symbol=symbol, meta_data=meta_market, date='20190621')
# fb = HistoricalEodIEX('FB', '20190715', meta_market)
