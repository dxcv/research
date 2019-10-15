import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quandl
import seaborn as sns


def plot_yield_curve(dates=None, ctry='US'):
    # prepare quandl query including auth_token, query data, write to df
    if dates is None:
        dates = ['2017-12-29']
    start_date = dates[0]
    end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    query, yields = None, None
    auth_token = 'WgALsBb8T2xgwghZaXiJ'

    if ctry == 'US':
        query = 'USTREASURY/YIELD'
        yields = quandl.get(query, authtoken=auth_token, start_date=start_date, end_date=end_date)
    elif ctry == 'CH':
        query = 'SNB/RENDOBLIM'
        yields = quandl.get(query, authtoken=auth_token)

    df = pd.DataFrame(columns=yields.columns)  # df of yields placeholder
    latest = yields.iloc[-1].name.to_pydatetime().date().strftime("%Y-%m-%d")  # most recent date
    day_before = yields.iloc[-2].name.to_pydatetime().date().strftime("%Y-%m-%d")  # previous date

    # append dates list with relevant dates: EOY 2017, 2018, most recent
    dates.append(day_before)
    dates.append('2018-12-31')
    dates.append(latest)
    dates = sorted(dates)
    for date in dates:
        date_series = yields.loc[date]
        df.loc[date] = date_series

    # spreads time series
    spreads = pd.DataFrame(index=yields.index, columns=['10y-2y', '10y-3m'])
    spreads['10y-2y'] = yields['10 YR'] - yields['2 YR']
    spreads['10y-3m'] = yields['10 YR'] - yields['3 MO']

    # plotting
    with plt.style.context('seaborn-whitegrid'):
        # instantiate figure including layout
        # cmap = sns.cubehelix_palette(rot=-.4, as_cmap=True)
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle("Yield Curve Data " + ctry + " for " + str(end_date), size=16)
        layout = (2, 2)
        year_on_year_ax = plt.subplot2grid(layout, (0, 0), fig=fig)
        one_day_change_ax = plt.subplot2grid(layout, (0, 1), fig=fig)
        one_day_change_bar_ax = plt.subplot2grid(layout, (1, 0), fig=fig)
        ytd_spreads_ax = plt.subplot2grid(layout, (1, 1), fig=fig)

        # yr on yr plot
        year_on_year_ax.set_xticks(np.arange(df.columns.shape[0]))
        year_on_year_ax.set_xticklabels(df.columns)
        df.iloc[[0, df.index.get_loc('2018-12-31'), -1]].transpose().plot(
            ax=year_on_year_ax, legend=True, title='Year End', style=['-^', '-*', '-<'],
            cmap=sns.cubehelix_palette(rot=-.2, as_cmap=True))

        # one day change yield curve
        one_day_change_ax.set_xticks(np.arange(df.columns.shape[0]))
        one_day_change_ax.set_xticklabels(df.columns)
        df.iloc[-2:].transpose().plot(ax=one_day_change_ax, legend=True, title='Latest Data', style=['-^', '-*'],
                                      cmap=sns.cubehelix_palette(rot=-.2, as_cmap=True))

        # one day change bar plot
        changes = pd.DataFrame(data=(df.iloc[-1] - df.iloc[-2]).values, index=df.columns, columns=['1 day change'])
        changes['YTD change'] = df.iloc[-1] - df.loc['2018-12-31']
        one_day_change_bar_ax.set_xticks(np.arange(df.columns.shape[0]))
        one_day_change_bar_ax.set_xticklabels(df.columns)
        changes.plot.bar(ax=one_day_change_bar_ax, title='Yield Change',
                         cmap=sns.cubehelix_palette(rot=-.2, as_cmap=True))
        one_day_change_bar_ax.axhline(alpha=0.3, color='black')

        # spreads plot
        spreads.plot(ax=ytd_spreads_ax, title='Spreads Time Series', cmap=sns.cubehelix_palette(rot=-.2, as_cmap=True))
        ytd_spreads_ax.axhline(y=0, color='black', alpha=0.3)

        plt.tight_layout()
        sns.despine()
    return None
