from classes.YieldCurve import YieldCurve

import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# yc = YieldCurve(ctry='US')
# yc_data = yc.load_yield_curve_data()


def get_dates_for_yield_curve_chart(dates):
    # dates as a datetime index
    most_recent_date = dates[-1].date()
    next_to_most_recent_date = dates[-2].date()
    wtd_date = (most_recent_date - pd.Timedelta(days=most_recent_date.weekday() + 3))# by construction a Friday
    ytd_date = (most_recent_date - pd.tseries.offsets.YearEnd(1)).date()
    mtd_date = (most_recent_date - pd.tseries.offsets.BMonthEnd(1)).date()

    date_index = pd.DatetimeIndex([ytd_date, mtd_date, wtd_date, next_to_most_recent_date, most_recent_date])
    return date_index


def prepare_yc_data_for_plot(data):
    # get dates index
    dates = get_dates_for_yield_curve_chart(data.index)
    data_for_chart = data.loc[dates]
    data_for_chart.index = data_for_chart.index.date

    return data_for_chart


def plotly_yc_chart(data, ctry, write_to_disk=False):
    # cut down data to size
    yc_chart_data = prepare_yc_data_for_plot(data)
    # individual data points as series: wtd, most recent, mtd, ytd
    current_yc = yc_chart_data.iloc[-1].copy()
    one_day_offset_yc = yc_chart_data.iloc[-2].copy()
    ytd_offset_yc = yc_chart_data.iloc[0].copy()
    wtd_offset_yc = yc_chart_data.iloc[2].copy()
    mtd_offset_yc = yc_chart_data.iloc[1].copy()

    # 10yr vs 2yr
    ten_vs_two_spread = data['10yr'] - data['2yr']
    ten_vs_3_mo_spread = data['10yr'] - data['3mo']

    # delta yc for different intervals
    wtd_change = current_yc - wtd_offset_yc
    wtd_change.name = 'WTD change'
    mtd_change = current_yc - mtd_offset_yc
    mtd_change.name = 'MTD change'
    ytd_change = current_yc - ytd_offset_yc
    ytd_change.name = 'YTD change'

    # figure
    fig = make_subplots(rows=2, cols=2, subplot_titles=['Year End', 'Latest Data', 'Yield Changes', 'Spreads'])
    # eoy
    fig.add_trace(go.Scatter(x=ytd_offset_yc.index, y=ytd_offset_yc, name=ytd_offset_yc.name.strftime('%Y-%m-%d'),
                             mode='lines+markers',  marker=dict(symbol="diamond", size=6)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=current_yc.index, y=current_yc, name=current_yc.name.strftime('%Y-%m-%d'),
                             mode='lines+markers',  marker=dict(symbol="diamond", size=6)),
                  row=1, col=1)

    # add current yc
    fig.add_trace(go.Scatter(x=current_yc.index, y=current_yc, name=current_yc.name.strftime('%Y-%m-%d'),
                             mode='lines+markers',  marker=dict(symbol="diamond", size=6)),
                  row=1, col=2)
    # add most recent yc
    fig.add_trace(go.Scatter(x=one_day_offset_yc.index, y=one_day_offset_yc, name=one_day_offset_yc.name.strftime(
        '%Y-%m-%d'),  mode='lines+markers',  marker=dict(symbol="diamond", size=6)),
                  row=1, col=2)

    # spread ts
    fig.add_trace(go.Scatter(x=ten_vs_two_spread.index, y=ten_vs_two_spread, name='10-2 yr spread',
                             mode='lines', marker=None), row=2, col=2)
    fig.add_trace(go.Scatter(x=ten_vs_3_mo_spread.index, y=ten_vs_3_mo_spread, name='10yr-3mo spread',
                             mode='lines', marker=None), row=2, col=2)

    # bar-chart changes
    fig.add_trace(go.Bar(x=wtd_change.index, y=wtd_change, name='WTD changes'), row=2, col=1)
    fig.add_trace(go.Bar(x=mtd_change.index, y=mtd_change, name='MTD changes'), row=2, col=1)
    fig.add_trace(go.Bar(x=ytd_change.index, y=ytd_change, name='YTD changes'), row=2, col=1)
    # show figure
    fig.update_layout(showlegend=True, template='seaborn', title='YC data {} for {}'.format(ctry,
        current_yc.name.strftime('%Y-%m-%d')))
    fig.show(renderer='browser')
    if write_to_disk is True:
        fig.write_image("C:/Users/28ide/PycharmProjects/strategy_research/plots/plotly/fig1.svg")


def plot_latest_yc(ctry='US'):
    yc = YieldCurve(ctry)
    yc_data = yc.load_yield_curve_data()
    # re-assign data for chart
    plotly_yc_chart(yc_data, ctry)


