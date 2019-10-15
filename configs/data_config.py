from configs.universe_spec import *
from utils.dataset import DataSet


def set_data_config(univ):
    """
    sets data specs including path, symbols, trading calendar
    :param univ: name of universe (e.g. universe_tech)
    :return data_config: config for data
    """
    data_config = DataSet()
    data_config.meta_data_root = './meta_data/universe_meta_data.json'
    data_config.symbols_list = univ

    return data_config
