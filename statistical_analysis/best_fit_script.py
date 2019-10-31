import numpy as np
import scipy.stats as st

from configs.universe_spec import universe_tech
from data_loading.load_from_disk.load_equities_data import load_equities_data_from_disk
from statistical_analysis.distribution_fitting import make_pdf, best_fit_distribution

# load data

eq = load_equities_data_from_disk(universe_tech, frequency='B')
prices = eq.get_cross_sectional_view('close')

# nflx example
nflx_ret = np.log(prices.NFLX).diff().dropna()

# get best fit, best is a tuple, with the name of the best dist (by SSE) and the params of the dist
best = best_fit_distribution(nflx_ret.dropna(), bins=12)
# get best distribution object
best_dist = getattr(st, best[0])
# feed into making a pdf to plot
pdf = make_pdf(best_dist, best[1])
ax = pdf.plot(lw=2, label='PDF', legend=True)
nflx_ret.dropna().plot(kind='hist', bins=12, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)
