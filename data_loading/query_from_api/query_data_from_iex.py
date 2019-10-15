from classes.IEX import HistoricalIntradayIEX, HistoricalEodIEX
from classes.Universe import Universe


def query_and_write_eod_data_for_symbols_from_iex(symbol, freq, date, attribute):
    """
    :param symbol: list of symbols for which to query data
    :param freq: frequency: either EOD or intraday (add further frequencies later)
    :return:
    """
    universe = Universe(symbol, frequency=freq)
    universe_meta_data = universe.read_meta_data_for_universe()
    for symb in symbol:
        meta_data_market = universe_meta_data.loc[symb]
        data = HistoricalEodIEX(symbol=symb, meta_data=meta_data_market, date=date)
        data.request_time_series_max(attribute=attribute)
        data.write_data_to_disk()
    return
