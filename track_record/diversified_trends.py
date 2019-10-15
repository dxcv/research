import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdate
import matplotlib


from utils.resampling_and_risk_metrics import calculate_drawdown_series


def create_dt_summary_plot():
    cta = pd.read_excel('C:/Users/28ide/OneDrive/Dokumente/Admin/Track Record/Data/multi-asset-chart-data.xlsx',
                           sheet_name='cta_chart_data', index_col=0)
    import seaborn as sns
    fig = plt.figure()
    font = {'size': 6}
    matplotlib.rc('font', **font)
    layout = (2, 3)

    # calculate stats
    cta_data = cta.copy()
    cta_data.columns = ['1741 Diversified Trends', 'MAN AHL Trend Alt.', 'Aspect Diversified Trends',
                        'Graham Capital Sys. Macro', 'AQR Managed Futures', 'SG CTA Index']
    cta_data = cta_data/cta_data.iloc[0]
    cta_data = cta_data.sort_values(cta_data.last_valid_index(), axis=1, ascending=False)
    returns_monthly = np.log(cta_data.resample('1M').last()).diff(1)

    roll_vol = returns_monthly.rolling(12).apply(lambda x: x.std() * np.sqrt(12), raw=False)
    mean_ret_ann = returns_monthly.mean() * 12
    std_ann = (returns_monthly.std() * np.sqrt(12))
    sr = mean_ret_ann / std_ann

    # results dataframe
    dt_stats = pd.DataFrame(columns=['mean', 'std', 'Sharpe'], index=cta_data.columns)
    dt_stats['mean'] = mean_ret_ann
    dt_stats['std'] = std_ann
    dt_stats['Sharpe'] = sr

    ax_nav = plt.subplot2grid(layout, (0, 0), colspan=2, fig=fig)
    ax_dd = plt.subplot2grid(layout, (0, 2), fig=fig)
    ax_dd.set_ylim(-0.5, 0)
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
    nav_list = [cta_data[fund] for fund in cta_data]

    with plt.style.context('seaborn-white'):
        my_fmt = DateFormatter("%Y")
        sns.set_palette("Greys_d")

        # plots nav
        sns.lineplot(data=nav_list, ax=ax_nav, legend='brief')


        # plots drawdown
        dd_list = [calculate_drawdown_series(cta_data[c]) for c in cta_data.columns]
        #
        sns.lineplot(data=dd_list, ax=ax_dd, legend='brief')
        #
        # calculate yearly returns (including first non-full year)
        ann_ret_list = []
        for column in cta_data.columns:
            df = pd.DataFrame(columns=['Return', 'Strategy', 'Date'])
            df['Return'] = np.log(cta_data[column]).diff(1).resample('A').sum().values
            df['Strategy'] = column
            df['Date'] = np.log(cta_data[column]).diff(1).resample('A').sum().index.strftime('%Y')
            ann_ret_list.append(df)
        ann_ret_df = pd.concat(ann_ret_list)
        ann_ret_df = ann_ret_df[ann_ret_df.Date != '2009']
        sns.barplot(y=ann_ret_df['Return'], x=ann_ret_df.Date, hue=ann_ret_df.Strategy, ax=ax_yr_ret)
        #ax_yr_ret.legend(fontsize='xx-small')

        # sr plot
        sns.barplot(y=dt_stats.Sharpe.index, x=dt_stats.Sharpe, orient='h', ax=ax_bar_sharpe)

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
        for ax in [ax_nav]:
            locator = mdate.MonthLocator([6,12])
            ax.xaxis.set_major_locator(locator)

            ax.xaxis.set_major_formatter(DateFormatter("%Y/%m"))

        for ax in [ax_dd]:
            locator = mdate.YearLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(DateFormatter('%Y'))

        sns.despine()
        filename = 'C:/Users/28ide/OneDrive/Dokumente/Tex Files/images/dt.pdf'
        plt.savefig(fname=filename, format='pdf', papertype='a4', orientation='portrait')