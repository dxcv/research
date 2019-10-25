import pandas as pd
from pandas.tseries import offsets as time_offset

from utils.dataset import DataSet


class WalkForwardBtCompounding(object):

    # -----------------------
    # initialize object
    # -----------------------

    def __init__(self, weights, prices, aum, **kwargs):
        """
        Note: functions and attributes prepended with "_" are "private"
        :param weights: dataframe of trading_signals / weights
        :param prices: dataframe of prices (equal to or longer than weights dates)
        :param aum: scalar with initial aum
        :param kwargs: # optional arguments: max_gross_leverage, max_net_leverage, name, dividends, contract_value, tc
        """

        # set aum
        self.aum = aum  # supplied (scalar)
        # all_leverage is a Boolean with True if weights do not need to scale to 1, else False
        # leverages
        if 'max_gross_leverage' in kwargs:
            self.max_gross_leverage = kwargs['max_gross_leverage']
        else:
            self.max_gross_leverage = 1
        if 'max_net_leverage' in kwargs:
            self.max_net_leverage = kwargs['max_net_leverage']
        else:
            self.max_net_leverage = 1
        if 'max_pos_leverage' in kwargs:
            self.max_pos_leverage = kwargs['max_pos_leverage']
        else:
            self.max_pos_leverage = 1
        if 'constant_exposure' in kwargs:
            self.constant_exposure = kwargs['constant_exposure']
            self.constant_long = 1
            self.constant_short = -1
        else:
            self.constant_exposure = False
        if 'config' in kwargs:
            self.max_pos_leverage = kwargs['config'].position_limit
            self.max_net_leverage = kwargs['config'].max_net_leverage
            self.max_gross_leverage = kwargs['config'].max_gross_leverage

        # weights
        self.weights = weights.loc[weights.first_valid_index():]  # trading weights/signals (supplied)
        self.scaled_weights = None  # scaled weights
        # dates
        self.trading_dt_index = None  # trading dates index
        self.price_date_index = None  # prices index
        self.bt_dt_index = None  # backtest date index
        # construct dates dataframe, index is date t, columns are dates t-1 and t+1
        self.dt_df = None
        self.frequency = None

        # prices
        self.close = prices
        if 'open' in kwargs:
            self.open = kwargs['open']  # open prices (supplied or none)
        else:
            self.open = pd.DataFrame(prices).shift(1)

        # further optional arguments
        if 'dividends' in kwargs:
            self.dividends = kwargs['dividends']  # dividends (supplied or none)
        else:
            self.dividends = pd.DataFrame(index=self.close.index, columns=self.close.columns).fillna(0)
        if 'trading_costs' in kwargs:
            self.trading_costs = kwargs['trading_costs']  # (supplied or none)
        else:
            self.trading_costs = pd.DataFrame(index=self.close.index, columns=self.close.columns).fillna(0)
        if 'point_value_instrument' in kwargs:
            self.point_value_instrument = kwargs['point_value_instrument']
        else:
            self.point_value_instrument = pd.DataFrame(index=self.close.index, columns=self.close.columns).fillna(1)

        # initialize holdings and pnl per instrument dataframes
        self.holdings = None  # initialize holdings
        self.pnl_per_instrument = None

        # backtest will be a dataset with entries net_asset_value (a pd.Series) and holdings (a pd.Dataframe)
        self._backtest = None

        # intialize values
        self._initialize_values()  # calculates attributes

    # ----------------------
    # initialize values and construct dt indices
    # ----------------------
    def _initialize_values(self):
        self._scale_weights()
        self._initialize_date_information()
        self._initialize_holdings()
        self._initialize_pnl_per_instrument()

    def _initialize_date_information(self):
        self.trading_dt_index = self.weights.index  # trading dates index
        self.price_date_index = self.close.index
        self.frequency = self.close.index.freq  # frequency of backtest
        self.bt_dt_index = self._construct_bt_dt_index()  # backtest date index
        # construct dates dataframe, index is date t, columns are dates t-1 and t+1
        self.dt_df = self._construct_dates_df()
        self.frequency = self.close.index.freq  # frequency of backtest

    # -----------------------
    # from signal to weight
    # -----------------------

    def _weight_scaling(self, x):

        if x.abs().sum() > self.max_gross_leverage:
            x /= (x.abs().sum() / self.max_gross_leverage)
            if abs(x.sum()) > self.max_net_leverage:
                x /= (x.abs().sum() / self.max_net_leverage)
        return x

    def _weight_scaling_alternative(self, x):
        index_order = x.index
        if self.constant_exposure is False:
            x_long = x[x >= 0].copy()
            x_short = x[x < 0].copy()
            position_limit = self.max_pos_leverage
            # ensure position limits are met
            x_long[x_long > position_limit] = position_limit
            x_short[x_short < position_limit * -1] = position_limit * -1
            gross_leverage = x_long.sum() + x_short.abs().sum()
            # ensure gross leverage is met
            if gross_leverage > self.max_gross_leverage:
                x_long /= (gross_leverage / self.max_gross_leverage)
                x_short /= (gross_leverage / self.max_gross_leverage)
            # ensure net leverage is met
            # logic if net leverage too high, reduce respective side to maximum allowed
            # pro rata / proportional leverage reduction on side where the leverage is too high
            net_leverage = (x_long.sum() + x_short.sum())
            # todo: min net leverage???
            if abs(net_leverage) > self.max_net_leverage:
                # pos or negative
                difference = net_leverage - self.max_net_leverage
                if difference > 0:
                    max_side = x_short.abs().sum() + self.max_net_leverage
                    x_long /= (x_long.sum() / max_side)
                else:
                    min_side = (self.max_net_leverage + x_long.sum()) * - 1
                    x_short /= (x_short.sum() / min_side)

            x = pd.concat([x_long, x_short], axis=0).reindex(index_order)
        else:
            x_long = x[x >= 0].copy()
            x_long = x_long / (x_long.sum() / self.constant_long)
            x_short = x[x < 0].copy()
            x_short = x_short / (x_short.sum() / self.constant_short)
            x = pd.concat([x_long, x_short], axis=0).reindex(index_order)
        return x

    def _scale_weights(self):
        # ensures that we are never invested with leverage > leverage_max,
        scaled_weights_tmp = self.weights.copy()
        scaled_weights = scaled_weights_tmp.apply(lambda x: self._weight_scaling_alternative(x), axis=1)
        self.scaled_weights = scaled_weights

    # ----------------------------------------------------------------
    # initialize  holdings, and pnl per instrument dataframes
    # ----------------------------------------------------------------

    # initialize holdings dataframe, first entry is at t=0 == 0
    def _initialize_holdings(self):
        holdings = pd.DataFrame(index=self.bt_dt_index, columns=self.weights.columns)
        holdings.iloc[0] = 0  # holdings initialized at 0
        self.holdings = holdings

    # initialize pnl per instrument dataframe, first entry is 0
    def _initialize_pnl_per_instrument(self):
        pnl_per_instrument = pd.DataFrame(index=self.bt_dt_index, columns=self.close.columns)
        pnl_per_instrument.iloc[0] = 0
        self.pnl_per_instrument = pnl_per_instrument

    # -------------------------------
    # prepare date index and df
    # -------------------------------

    def _construct_bt_dt_index(self):
        """ constructs the t0 dates index that runs from t0 to T
        if the price series is longer than the weights series use the the l
        The function takes the weights index and prepends it either with the last available previous date from the
        price index or else prepends t1 with a timedelta """
        dt_t0_tmp = self.price_date_index.copy()
        # where is first date
        first_weight_date = self.trading_dt_index[0]

        if dt_t0_tmp[0] < first_weight_date:
            # prices start before first weight date, bt index starts at date closest to weight date start
            # initialization date t0 is the date closest to the date of the first weight t1
            initialization_date_index = dt_t0_tmp.get_loc(first_weight_date) - 1
            dates_t0_index = dt_t0_tmp[initialization_date_index:]

        else:
            freq = self.frequency
            if freq == 'B':
                initialization_date = dt_t0_tmp[0] - time_offset.BDay(1)
            elif freq == 'D':
                initialization_date = dt_t0_tmp[0] - time_offset.Day(1)
            elif freq == 'min':
                initialization_date = dt_t0_tmp[0] - time_offset.Minute(1)
            elif freq == 'H':
                initialization_date = dt_t0_tmp[0] - time_offset.Hour(1)
            else:
                import pdb
                pdb.set_trace()
                assert freq == 'S'
                initialization_date = dt_t0_tmp[0] - time_offset.Second(1)
            # prepend index with "artificial" first datetime; interval chosen to match frequency of price index
            dates_t0_index = dt_t0_tmp.append(pd.DatetimeIndex([initialization_date])).sort_values()
        return dates_t0_index

    def _construct_dates_df(self):
        """ constructs a dataframe of dates with index t, and columns dt_previous and dt_next for t-1 and t + 1
        we need date t + 1 to calculate holdings at t, we buy/sell a position at the open of t + 1
        we need date t - 1 to calculate holdings at t if t is not a trading date and to calculate the current
        investment value which is the investment value at t-1 + the pnl over period [t-1, t] """
        dt_df = pd.DataFrame(index=self.bt_dt_index)
        dt_df['dt_previous'] = dt_df.index.copy()
        # date at t-1 is the previous date for date t
        dt_df.dt_previous = dt_df.dt_previous.shift(1)
        # first date previous is initialization date
        dt_df.iloc[0].previous = self.bt_dt_index[0]
        dt_df['dt_next'] = dt_df.index.copy()
        dt_df.dt_next = dt_df['dt_next'].shift(-1)
        dt_df['dt_now'] = dt_df.index.copy()

        # add most recent trading dates, we need this to check whether weights at the most recent trading date were 0
        # holdings at trading date
        # dt_df['most_recent_trading_date'] = pd.NaT
        # dt_df['most_recent_trading_date'].loc[self.trading_dt_index] = self.trading_dt_index
        # dt_df.most_recent_trading_date.ffill(inplace=True)

        return dt_df.iloc[1:]  # we don't need the first date entry

    # -------------------------
    # instrument pnl
    # -------------------------

    def _calculate_instrument_pnl_trading_date(self, date):
        """ calculates pnl for period t-1 to t from holdings at t-1. As date t-1 is a trading date the pnl per position
        calculated as close at t - open t """
        pnl_instrument = self.close.loc[date] - self.open.loc[date] + self.dividends.loc[date]
        return pnl_instrument

    def _calculate_instrument_pnl_non_trading_date(self, date):
        """ pnl over period [t-1,t] is calculated as close_t - close_t-1 """
        pnl_instrument = self.close.diff(1, axis=0).loc[date] + self.dividends.loc[date]
        return pnl_instrument

    def _calculate_instruments_pnl(self, date_series):
        """ date is t """
        previous_date = date_series.dt_previous
        now = date_series.dt_now
        if previous_date in self.trading_dt_index:
            pnl_instruments = self._calculate_instrument_pnl_trading_date(now)
        else:
            pnl_instruments = self._calculate_instrument_pnl_non_trading_date(now)
        return pnl_instruments

    # -------------------
    # holdings
    # -------------------

    def _update_holdings_trading_date(self, dates_series, net_asset_value_now):
        """
        update date holdings when
        :param dates_series:
        :param net_asset_value_now: scalar
        :return:
        """
        now = dates_series.dt_now
        next_date = dates_series.dt_next
        holdings = self.scaled_weights.loc[now] * net_asset_value_now / (
                    self.open.loc[next_date] * self.point_value_instrument.loc[now])
        return holdings

    def _update_holdings(self, dates_series, holdings_previous, net_asset_value_now):
        """
        :param dates_series: pd.Series of dates, index=now, columns=[date_previous, date_next]
        :return:
        """
        current_date_is_not_a_trading_date = dates_series.dt_now not in self.trading_dt_index

        # if not a trading date new holdings == old holdings
        if current_date_is_not_a_trading_date:
            holdings = holdings_previous
        else:
            # check whether trading weights are all 0
            new_weights_are_zero = self.scaled_weights.loc[dates_series.dt_now].sum() == 0
            if new_weights_are_zero:
                holdings = pd.Series(index=[dates_series.dt_now]).fillna(0)
            else:
                holdings = self._update_holdings_trading_date(dates_series, net_asset_value_now)
        return holdings

    def _get_pnl_attribution(self):
        cumulative_pnl_per_instrument = (self.holdings * self.pnl_per_instrument).cumsum().dropna()

        return cumulative_pnl_per_instrument

    # -----------------
    # NAV
    # -----------------

    def _update_net_asset_value(self, dates_series, net_asset_value_previous, holdings_previous):
        """
        :param dates_series: pd.Series of date, index=now, columns=date_previous, and date_nex
        :param net_asset_value_previous: scalar of t-1 net asset value
        :param holdings_previous: scalar of t-1 holdings
        :return:
        """
        point_value_now = self.point_value_instrument.loc[dates_series.dt_now]
        dividends_now = self.dividends.loc[dates_series.dt_now]
        pnl_now = self._calculate_instruments_pnl(dates_series)

        net_asset_value_now = net_asset_value_previous + (
                holdings_previous * point_value_now * (pnl_now + dividends_now)).sum()
        return net_asset_value_now

    # --------------------
    # RUN BACKTEST
    # --------------------

    def calculate_backtest_performance(self):
        """
        :return: returns a DataSet object with 2 keys: net_asset_value (pd.Series), holdings (pd.DataFrame)
        """
        net_asset_value_series = pd.Series()
        net_asset_value_series.loc[self.bt_dt_index[0]] = self.aum
        holdings_df = self.holdings

        for date in self.dt_df.index:
            # get previous date
            date_previous = self.dt_df.dt_previous.loc[date]
            # calculate pnl per instrument
            self.pnl_per_instrument.loc[date] = self._calculate_instruments_pnl(self.dt_df.loc[date])
            net_asset_value_now = self._update_net_asset_value(
                self.dt_df.loc[date], net_asset_value_series.loc[date_previous], holdings_df.loc[date_previous])
            # update nav at date t
            net_asset_value_series.loc[date] = net_asset_value_now
            # don't update holdings at last date as we have no open t+1 price
            if date != self.dt_df.index[-1]:
                holdings_df.loc[date] = self._update_holdings(
                    self.dt_df.loc[date], holdings_df.loc[date_previous], net_asset_value_now)
            if date.is_year_end and date.time().hour is 23:
                print("\nStrategy value for date " + str(date) + " for strategy calculated \n")
        backtest_dataset = DataSet()
        backtest_dataset.holdings = holdings_df
        backtest_dataset.net_asset_value = net_asset_value_series.resample(self.frequency).last()
        backtest_dataset.cumulative_pnl_per_instrument = self._get_pnl_attribution()
        self._backtest = backtest_dataset

        return self._backtest

    @property
    def backtest(self):
        if self._backtest is None:
            self.calculate_backtest_performance()
        return self._backtest


class WalkForwardBtNoCompounding(WalkForwardBtCompounding):
    """
    subclass of WalkForwardBtCompounding. When calculating the new holdings we calculate on the basis of the initial
    aum rather than of the previous NAV, i.e. we do not compound. At each new date we invest the initial aum, not the
    current NAV
    """

    def __init__(self, weights, prices, aum, **kwargs):
        super().__init__(weights, prices, aum, **kwargs)

    def calculate_backtest_performance(self):

        # outputs a dataset with entries a) net_asset_value (a series) and b) holdings: a holdings dataframe
        net_asset_value_series = pd.Series()
        net_asset_value_series.loc[self.bt_dt_index[0]] = self.aum
        holdings_df = self.holdings

        for date in self.dt_df.index:
            # get previous date
            date_previous = self.dt_df.dt_previous.loc[date]
            # calculate pnl per instrument
            self.pnl_per_instrument.loc[date] = self._calculate_instruments_pnl(self.dt_df.loc[date])
            net_asset_value_now = self._update_net_asset_value(
                self.dt_df.loc[date], net_asset_value_series.loc[date_previous], holdings_df.loc[date_previous])
            # update nav at date t
            net_asset_value_series.loc[date] = net_asset_value_now
            # don't update holdings at last date as we have no open t+1 price
            # if date != self.trading_dt_index[-1]:
            if date != self.dt_df.index[-1]:
                holdings_df.loc[date] = self._update_holdings(
                    self.dt_df.loc[date], holdings_df.loc[date_previous], self.aum)
            if date.is_year_end and date.time().hour is 23:
                print("\nStrategy value for date " + str(date) + " for strategy calculated \n")
        backtest_dataset = DataSet()
        backtest_dataset.holdings = holdings_df
        backtest_dataset.net_asset_value = net_asset_value_series.resample(self.frequency).last()
        self._backtest = backtest_dataset

        return self._backtest
