import datetime as dt

import numpy as np
import pandas as pd

from data_loading.load_from_disk.load_equities_data import load_equities_data_from_disk

from statsmodels.tsa.stattools import adfuller
from plots.time_series_plots import time_series_plot


# load tech data from meta file and IEX
# todo: download full universe tech data from IEX
universe_tech = ['FB', 'AAPL']
tech_data = load_equities_data_from_disk(universe_tech, frequency='B')
close_data = tech_data.get_cross_sectional_view('close')
aapl = close_data['FB'].copy()
amzn = close_data['AAPL'].copy()

# todo: loop through all possible pairs, run adf fuller, set profit taking/stop-loss limits, find entry and exit points

# estimate beta
beta = np.exp((np.log(amzn)-np.log(aapl)).mean())

# estimate us
us = np.log(amzn)-np.log(aapl)-np.log(beta)

# adf
adf_ex = adfuller(us.dropna(), maxlag=10, regression='c', autolag='AIC', regresults=True)

# time series plot
time_series_plot(us.dropna(), 10)

