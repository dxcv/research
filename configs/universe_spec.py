from utils.dataset import DataSet


universe = DataSet()

# ---------------------------------

universe_tech = ['AAPL', 'AMZN', 'FB', 'GOOG', 'NFLX', 'TSLA']


# ---------------------------------

universe_etf = ['GLD', 'WTMF']

# ---------------------------------

universe_crypto = ['BTCUSD', 'LTCUSD', 'ETHUSD']

# read from json: pd.read_json(root) give back df with symbols as columns and properties as indices

# config per market: read meta data from json, create a data set per symbol with all the meta data.
