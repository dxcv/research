from functools import partial
import numpy as np
import pandas as pd


def spline_series(df, spline):
    splined_series = pd.Series(spline(df), index=df.index)
    return splined_series


cta_spline = partial(np.interp, xp=[-3., -2.5, -2., -1.5, 0., 1.5, 2., 2.5, 3.],
                         fp=[-1., -1., -1.5, -1.5, 0., 1.5, 1.5, 1., 1.])

grd_spline = partial(np.interp, xp=[-2., -1., 0., 1., 1.5, 2.0],
                                     fp=[0.25, 0.25, 1., 1., 1.25, 1.25])