import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scs
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt


def calculate_signal_quantiles(xy_df, bins=5):
    xy_df = xy_df.assign(signal_quantiles=pd.qcut(xy_df.signals, q=bins))
    xy_df = xy_df.assign(signal_bins=pd.cut(xy_df.signals, bins=bins))
    return xy_df


def time_series_plot(y, lags=None, fig_size=(11.7, 8.27), title='ts plot'):
    """
    :param y: time series pd.Series
    :param lags: number of lags to be plotted
    :param fig_size: size of figure
    :param title: title of chart
    :return:
    """

    with plt.style.context('seaborn-whitegrid'):
        # instantiate figure including layout
        fig = plt.figure(figsize=fig_size)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2, fig=fig)
        acf_ax = plt.subplot2grid(layout, (1, 0), fig=fig)
        pacf_ax = plt.subplot2grid(layout, (1, 1), fig=fig)
        qq_ax = plt.subplot2grid(layout, (2, 0), fig=fig)
        pp_ax = plt.subplot2grid(layout, (2, 1), fig=fig)

        # plot data to plots
        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return


def xy_plot(xy_df, fig_size=(11.7, 8.27), title='xy plot'):
    """
    todo: winsorizing: xy_plot(xy_df['2015-01-15':].query('abs(returns) < 0.1'))
    :param xy_df: time series pd.DataFrame ('signal', 'return') shifted
    :param fig_size: size of figure
    :param title: title of chart
    :return:
    """

    with plt.style.context('seaborn-whitegrid'):
        # instantiate figure including layout
        fig = plt.figure(figsize=fig_size)
        fig.suptitle(title, size=16)
        layout = (2, 3)
        reg_ax = plt.subplot2grid(layout, (0, 0), colspan=2, fig=fig)
        box_ax = plt.subplot2grid(layout, (0, 2), fig=fig)
        box_ax_sigret = plt.subplot2grid(layout, (1, 0), fig=fig)
        hist_ax = plt.subplot2grid(layout, (1, 1), fig=fig)
        hist_ax_2 = plt.subplot2grid(layout, (1, 2), fig=fig)

        # regplot
        y = xy_df['returns'].copy()
        x = xy_df['signals'].copy()
        # plot data to plots
        sns.regplot(x, y, scatter=True, order=2, ax=reg_ax)
        reg_ax.set_title('polynomial fit')

        # boxplot
        # calculate quantiles and equally spaced bins
        xy_df = calculate_signal_quantiles(xy_df, bins=5)
        xy_df = xy_df.assign(signal_return=xy_df.signals * xy_df.returns)
        box_plot = sns.boxplot(x="signal_quantiles", y="returns", data=xy_df, ax=box_ax, palette="Blues")
        box_plot_sigret = sns.violinplot(x="signal_quantiles", y="signal_return", data=xy_df, ax=box_ax_sigret,
                                         palette="Blues")

        # plot and add annotation
        box_plot.set_title('signal vs return boxplot')
        box_plot_sigret.set_title('signal vs signalreturn boxplot')

        # add boxes with median values
        ax = box_plot.axes
        lines = ax.get_lines()
        categories = ax.get_xticks()
        for cat in categories:
            # every 4th line at the interval of 6 is median  line
            # boxes, medians, whiskers, caps, fliers, means
            y = round(lines[4 + cat * 6].get_ydata()[0], 4)
            ax.text(
                cat,
                y,
                f'{y}',
                ha='center',
                va='center',
                fontweight='bold',
                size=8,
                color='white',
                bbox=dict(facecolor='#445A64'))
        box_plot.figure.tight_layout()

        # hist
        sns.distplot(xy_df.signals, kde=True, ax=hist_ax)
        hist_ax.set_title('distribution signal')
        sns.distplot(xy_df.returns, kde=True, ax=hist_ax_2)
        hist_ax_2.set_title('distribution return')
        plt.tight_layout()

    return
