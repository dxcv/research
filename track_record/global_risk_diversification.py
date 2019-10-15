import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdate
import matplotlib


from utils.resampling_and_risk_metrics import calculate_drawdown_series


def create_grd_summary_plot():
    nav_global = pd.read_excel('C:/Users/28ide/OneDrive/Dokumente/Admin/Track Record/Data/multi-asset-chart-data.xlsx',
                               sheet_name='Sheet3', usecols='J:N', index_col=0, skipfooter=27)
    import seaborn as sns
    fig = plt.figure()
    font = {'size': 6}
    matplotlib.rc('font', **font)
    layout = (2, 3)

    # calculate stats
    nav_glo = nav_global.copy()
    nav_glo.columns = ['1741 GRD', 'Invesco Balanced Risk Allocation', 'Black Rock Market Advantage', 'AQR Risk Parity']
    returns_monthly = np.log(nav_glo.resample('1M').last()).diff(1)
    #returns_daily = np.log(nav_glo).diff(1)
    roll_vol = returns_monthly.rolling(12).apply(lambda x: x.std() * np.sqrt(12), raw=False)
    mean_ret_ann = returns_monthly.mean() * 12
    std_ann = (returns_monthly.std() * np.sqrt(12))
    sr = mean_ret_ann / std_ann

    # results dataframe
    grd_stats = pd.DataFrame(columns=['mean', 'std', 'Sharpe'], index=[
        '1741 GRD', 'Invesco Balanced Risk Allocation', 'Black Rock Market Advantage', 'AQR Risk Parity'])
    grd_stats['mean'] = mean_ret_ann
    grd_stats['std'] = std_ann
    grd_stats['Sharpe'] = sr

    ax_nav = plt.subplot2grid(layout, (0, 0), colspan=2, fig=fig)
    ax_dd = plt.subplot2grid(layout, (0, 2), fig=fig)
    ax_dd.set_ylim(-0.2, 0)
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
    nav_list = [nav_glo[fund] for fund in nav_glo]

    with plt.style.context('seaborn-white'):
        my_fmt = DateFormatter("%Y")
        sns.set_palette("Greys_d")

        # plots nav
        sns.lineplot(data=nav_list, ax=ax_nav, legend='brief')


        # plots drawdown
        dd_list = [calculate_drawdown_series(nav_glo[c]) for c in nav_glo.columns]
        #
        sns.lineplot(data=dd_list, ax=ax_dd, legend='brief')
        #
        # calculate yearly returns (including first non-full year)
        ann_ret_list = []
        for column in nav_glo.columns:
            df = pd.DataFrame(columns=['Return', 'Strategy', 'Date'])
            df['Return'] = np.log(nav_glo[column]).diff(1).resample('A').sum().values
            df['Strategy'] = column
            df['Date'] = np.log(nav_glo[column]).diff(1).resample('A').sum().index.strftime('%Y')
            ann_ret_list.append(df)
        ann_ret_df = pd.concat(ann_ret_list)
        ann_ret_df = ann_ret_df[ann_ret_df.Date != '2009']
        sns.barplot(y=ann_ret_df['Return'], x=ann_ret_df.Date, hue=ann_ret_df.Strategy, ax=ax_yr_ret)
        #ax_yr_ret.legend(fontsize='xx-small')

        # sr plot
        sns.barplot(y=grd_stats.Sharpe.index, x=grd_stats.Sharpe, orient='h', ax=ax_bar_sharpe)

        # roll vol
        sns.lineplot(data=roll_vol, ax=ax_rolling_vol, legend='brief')
        # a4 size
        fig.set_size_inches(11.7, 8.27)
        # fig.set_size_inches(8.27, 11.7)
        # all_axes = [ax_nav, ax_dd, ax_yr_ret, ax_rolling_vol]
        # for ax in all_axes:
        #     ax.legend(fontsize='xx-small')

        locator = mdate.YearLocator()
        ax_rolling_vol.xaxis.set_major_locator(locator)

        ax_rolling_vol.xaxis.set_major_formatter(DateFormatter("%Y"))

        sns.despine()


        filename = 'C:/Users/28ide/OneDrive/Dokumente/Tex Files/images/grd.pdf'
        plt.savefig(fname=filename, format='pdf', papertype='a4', orientation='portrait')

def create_at_summary_plot():
    # formatting %
    @ticker.FuncFormatter
    def major_formatter(x):
        return "[%.2f]" % x

    # read data
    data = pd.read_csv('C:/Users/28ide/OneDrive/Dokumente/Admin/Track Record/ACADTES_LX_Equity.csv', sep=',',
                       index_col=0, skiprows=2)
    sg_cta_index = pd.read_excel(
        'C:/Users/28ide/OneDrive/Dokumente/Admin/Track Record/SG_CTA_Index_historical_data.xls',
        skiprows=1, index_col=0)

    data.index = pd.to_datetime(data.index, format='%d.%m.%Y')
    data = pd.concat([data, sg_cta_index['VAMI']], axis=1, join='inner')
    data.columns = ['Adaptive Trends', 'SG CTA Index']
    indexed_perf = data / data.iloc[0]

    # calculate performance metrics
    returns_monthly = np.log(data.resample('1M').last()).diff(1)
    returns_daily = np.log(data).diff(1)
    mean_ret_ann = returns_monthly.mean() * 12
    std_ann = (returns_monthly.std() * np.sqrt(12))
    sr = mean_ret_ann / std_ann

    # results dataframe
    adaptive_trends_stats = pd.DataFrame(columns=['mean', 'std', 'sr'], index=['Adaptive Trends', 'SG CTA Index'])
    adaptive_trends_stats['mean'] = mean_ret_ann
    adaptive_trends_stats['std'] = std_ann
    adaptive_trends_stats['sr'] = sr

    dd_series = pd.DataFrame(columns=['Adaptive Trends', 'SG CTA Index'])
    for series in dd_series:
        dd_series[series] = calculate_drawdown_series(data[series])

    roll_vol = returns_daily.rolling(125).apply(lambda x: x.std() * np.sqrt(125))

    # plotting
    with plt.style.context('seaborn-white'):
        my_fmt = DateFormatter("%m/%d")
        sns.set_palette("Greys_d")
        # colors = ['tab:blue', 'tab:green']
        # create chart layout
        fig = plt.figure()
        # fig.suptitle('AC Adaptive Trends', size=16)
        layout = (2, 3)
        rebased_indx_ax = plt.subplot2grid(layout, (0, 0), colspan=2, fig=fig)
        drawdown_ax = plt.subplot2grid(layout, (0, 2), fig=fig)
        sharpe_ax = plt.subplot2grid(layout, (1, 0), fig=fig)
        ax_ann_ret = plt.subplot2grid(layout, (1, 1), fig=fig)
        ax_list = [rebased_indx_ax, drawdown_ax]
        ax_rol_vol = plt.subplot2grid(layout, (1, 2), fig=fig)

        # dates
        for ax in ax_list:
            ax.xaxis.set_major_formatter(my_fmt)

        # subtitles
        rebased_indx_ax.set_title('NAV Rebased vs. Benchmark')
        drawdown_ax.set_title('Drawdown')
        drawdown_ax.set_ylim(-0.2, 0)
        rebased_indx_ax.set_ylim(0.8, 1.2)
        sharpe_ax.set_title('Sharpe Ratio')
        sharpe_ax.set_xlim(0, 1)
        ax_ann_ret.set_title('Yr-on-Yr Returns')
        ax_rol_vol.set_title('Rolling 6-month Volatility')
        ax_rol_vol.set_ylim(0, 0.2)
        # mean-std-scatter
        # for strategy in range(len(labels)):
        #     adaptive_trends_stats.iloc[[strategy]].plot.scatter(
        #         x='std', y='mean', ax=sharpe_ax,
        #         legend=True, label=labels[strategy],
        #          s=200)

        # rebased index / dd series
        perf_list = [indexed_perf[c] for c in data.columns]
        dd_list = [calculate_drawdown_series(data[c]) for c in data.columns]

        sns.lineplot(data=perf_list, ax=rebased_indx_ax, legend='brief')
        sns.lineplot(data=dd_list, ax=drawdown_ax, legend='brief')
        # y_value = ['{:,.2f}'.format(x) + '%' for x in drawdown_ax.get_yticks()]
        # ax.set_yticklabels(y_value)
        sns.barplot(y=adaptive_trends_stats.sr.index, x=adaptive_trends_stats.sr, orient='h', ax=sharpe_ax)

        # ann ret plots
        ann_ret_list = []
        for column in data.columns:
            df = pd.DataFrame(columns=['return', 'Strategy', 'date'])
            df['return'] = np.log(data[column]).diff(1).resample('A').sum().values
            df['Strategy'] = column
            df['date'] = np.log(data[column]).diff(1).resample('A').sum().index.strftime('%Y')
            ann_ret_list.append(df)
        ann_ret_df = pd.concat(ann_ret_list)

        sns.barplot(y=ann_ret_df['return'], x=ann_ret_df.date, hue=ann_ret_df.Strategy, ax=ax_ann_ret)

        # ann voll roll
        sns.lineplot(data=roll_vol, ax=ax_rol_vol, legend='brief')
        # a4 size
        fig.set_size_inches(11.7, 8.27)
        sns.despine()

        # save chart to disk
        filename = 'C:/Users/28ide/OneDrive/Dokumente/Tex Files/images/grd.pdf'
        # plt.tight_layout()

        plt.savefig(fname=filename, format='pdf', papertype='a4', orientation='landscape')
    return


# ----------------------------------------
# performance and risk calculation functions
# ----------------------------------------

def calculate_yearly_returns():
    pass
