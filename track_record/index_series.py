import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdate
import matplotlib


from utils.resampling_and_risk_metrics import calculate_drawdown_series


def create_index_series_summary_plot():
    # data
    data_path = 'C:/Users/28ide/OneDrive/Dokumente/Admin/Track Record/Data/'
    spi_file = 'hspitr.csv'
    index_series_file = 'index_series.xls'
    index_series = pd.read_excel(data_path + index_series_file, usecols=[0, 1, 4, 7, 10, 13], index_col=0,
                                 names=['Momentum', 'Value', 'Quality', 'Risk Parity', 'Min Vol'], skiprows=2)

    index_series.index = pd.to_datetime(index_series.index)  # convert to datetime

    # load spi data
    tmp = pd.read_csv(data_path + spi_file, sep=';', skiprows=6, index_col=0, usecols=[0, 2])
    tmp.index = pd.to_datetime(tmp.index, format='%d.%m.%Y')
    spi = tmp.reindex(tmp.index.sort_values())
    spi.columns = ['SPI']

    # merge spi and index series data
    equities_merged_df = index_series.merge(spi, right_index=True, left_index=True)
    start_date_index_series = '2014-04-30'
    equities_cut_df = equities_merged_df[start_date_index_series:] / equities_merged_df[start_date_index_series:].iloc[
        0]
    equities_cut_df = equities_cut_df.sort_values(equities_cut_df.last_valid_index(), axis=1, ascending=False)

    # figure
    fig = plt.figure()
    font = {'size': 6}
    matplotlib.rc('font', **font)
    layout = (2, 3)

    # calculate stats
    index_series = equities_cut_df.copy()
    returns_monthly = np.log(index_series.resample('1M').last()).diff(1)

    roll_vol = returns_monthly.rolling(12).apply(lambda x: x.std() * np.sqrt(12), raw=False)
    mean_ret_ann = returns_monthly.mean() * 12
    std_ann = (returns_monthly.std() * np.sqrt(12))
    sr = mean_ret_ann / std_ann

    # results dataframe
    is_stats = pd.DataFrame(columns=['mean', 'std', 'Sharpe'], index=
        equities_cut_df.columns)
    is_stats['mean'] = mean_ret_ann
    is_stats['std'] = std_ann
    is_stats['Sharpe'] = sr

    ax_nav = plt.subplot2grid(layout, (0, 0), colspan=2, fig=fig)
    ax_dd = plt.subplot2grid(layout, (0, 2), fig=fig)
    ax_dd.set_ylim(-0.3, 0)
    ax_bar_sharpe = plt.subplot2grid(layout, (1, 0), fig=fig)
    ax_yr_ret = plt.subplot2grid(layout, (1, 1), fig=fig)
    ax_rolling_vol = plt.subplot2grid(layout, (1, 2), fig=fig)

    ax_nav.set_title('NAV Rebased vs. Benchmark')
    ax_dd.set_title('Drawdown')
    ax_bar_sharpe.set_title('Sharpe Ratio')
    ax_yr_ret.set_title('Yr-on-Yr Returns')
    ax_rolling_vol.set_title('Rolling 1-Yr Volatility')

    ax_nav.set(xlabel='Date', ylabel='NAV')
    ax_dd.set(xlabel='Date', ylabel='Drawdown')
    ax_rolling_vol.set(xlabel='Date', ylabel='1-Yr Volatility')
    ax_yr_ret.set(xlabel='Date', ylabel='Yr-on-Yr Return')

    # nav_trend = nav_trends.reindex(['1741 Div Trends', 'AHL Div Trends',
    #                                 'AQR MF', 'Graham Sys. Macro', 'Newedge CTA Index'],axis=1).copy()
    # nav_ada = nav_adaptive.reindex(['Adaptive Risk', 'AQR Risk Parity','Blackrock Market Ad.', 'Invesco Bal.'],
    #                                axis=1).copy()
    nav_list = [index_series[fund] for fund in index_series]

    with plt.style.context('seaborn-white'):
        my_fmt = DateFormatter("%Y")
        sns.set_palette("Greys_d")

        # plots nav
        sns.lineplot(data=nav_list, ax=ax_nav, legend='brief')


        # plots drawdown
        dd_list = [calculate_drawdown_series(index_series[c]) for c in index_series.columns]
        #
        sns.lineplot(data=dd_list, ax=ax_dd, legend='brief')
        #
        # calculate yearly returns (including first non-full year)
        ann_ret_list = []
        for column in index_series.columns:
            df = pd.DataFrame(columns=['Return', 'Strategy', 'Date'])
            df['Return'] = np.log(index_series[column]).diff(1).resample('A').sum().values
            df['Strategy'] = column
            df['Date'] = np.log(index_series[column]).diff(1).resample('A').sum().index.strftime('%Y')
            ann_ret_list.append(df)
        ann_ret_df = pd.concat(ann_ret_list)
        ann_ret_df = ann_ret_df[ann_ret_df.Date != '2009']
        sns.barplot(y=ann_ret_df['Return'], x=ann_ret_df.Date, hue=ann_ret_df.Strategy, ax=ax_yr_ret)
        #ax_yr_ret.legend(fontsize='xx-small')

        # sr plot
        sns.barplot(y=is_stats.Sharpe.index, x=is_stats.Sharpe, orient='h', ax=ax_bar_sharpe)

        # roll vol
        sns.lineplot(data=roll_vol, ax=ax_rolling_vol, legend='brief')
        # a4 size
        fig.set_size_inches(11.7, 8.27)

        # all_axes = [ax_nav, ax_dd, ax_yr_ret, ax_rolling_vol]
        # for ax in all_axes:
        #     ax.legend(fontsize='xx-small')

        locator = mdate.YearLocator()
        ax_rolling_vol.xaxis.set_major_locator(locator)

        ax_rolling_vol.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax_dd.xaxis.set_major_formatter(DateFormatter("%Y"))

        for ax in [ax_nav, ax_dd, ax_rolling_vol]:
            locator = mdate.YearLocator()
            ax.xaxis.set_major_locator(locator)

            ax.xaxis.set_major_formatter(DateFormatter("%Y"))


        sns.despine()


        filename = 'C:/Users/28ide/OneDrive/Dokumente/Tex Files/images/is.pdf'
        plt.savefig(fname=filename, format='pdf', papertype='a4', orientation='portrait')