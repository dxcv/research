from utils.dataset import DataSet


class DataContainer(object):
    # DataContainer has structure: key is symbol, value is data (pd.DataFrame)
    # idea: intialize data container with variable number of dataframes with symbols as names
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)




