from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from utils.date_utils import get_nth_previous_month, get_nth_previous_year, get_nth_weekday_of_contract_month
from utils.futures_utils import get_contract_month_from_code, get_last_trading_day_of_month_for_exchange, nearest
from utils.dataset import DataSet


class Future(object):

    def __init__(self, symbol, **kwargs):
        self.symbol = symbol
        if 'source' in kwargs:
            self.source = kwargs['source']
        else:
            # default source for futures data is CQG
            self.source = 'CQG'

        self.dates = None
        self.contracts = None

        self.data_path = Path('C:/Users/28ide/Data/Futures/futures_data/dailydata/')

        self.roll_rule = pd.read_excel('C:/Users/28ide/Data/Futures/futures_data/dailydata/futures_roll_rules.xlsx',
                                       index_col='Row')[symbol]

        self.meta_data = self._load_meta_data()

        self.raw_data = self._load_cqg_data_from_disk()
        if 'exchange' in kwargs:
            self.exchange = kwargs['exchange']
        else:
            self.exchange = self.meta_data.loc['Exchange']  # default exchange

        self.near_prices = None
        self.far_prices = None
        self.near_contracts = None
        self.far_contracts = None

    def _load_meta_data(self):
        meta_data_path = self.data_path.joinpath('meta.json')
        meta_data = pd.read_json(meta_data_path)[self.symbol]
        self.meta_data = meta_data
        return meta_data

    def _load_cqg_data_from_disk(self):
        # loads close data from disk
        cqg_prefix = 'F.US.'
        symbol = self.symbol
        raw_data = pd.DataFrame()
        date_format = '%Y%m%d'
        for entry in self.data_path.glob(cqg_prefix + symbol + '*'):
            contract_data = pd.read_csv(entry, index_col=1, names=['ticker', 'o', 'h', 'l', 'c'])
            contract_data.index = pd.to_datetime(contract_data.index, format=date_format)
            raw_data = pd.concat([raw_data, contract_data])
        raw_data.sort_index(inplace=True)
        self.raw_data = raw_data
        print('*** Read data for {} from disk ***'.format(self.symbol))

        raw_data_stripped = self._drop_inactive_contracts_from_raw_data()
        print('*** Keep active contracts only: {} ***'.format(self.roll_rule.active_months))

        self.dates = raw_data_stripped.index.drop_duplicates()
        self.contracts = raw_data_stripped.ticker.drop_duplicates()

        return raw_data_stripped

    def _drop_inactive_contracts_from_raw_data(self):
        raw_data = self.raw_data
        active_months = self.roll_rule.active_months
        months_list = list(active_months)
        tickers = raw_data.ticker

        def get_contract_month(string_symbol):
            no_prefix_no_symbol = string_symbol.replace('F.US.' + self.symbol, '')
            contract_month = no_prefix_no_symbol[0]
            return contract_month

        contract_months = tickers.apply(lambda x: get_contract_month(x))

        raw_data['contract_month'] = contract_months
        raw_data_active_months = raw_data[raw_data.contract_month.str.contains('|'.join(months_list))]
        self.raw_data = raw_data_active_months
        return raw_data_active_months

    def _get_contract_year_and_month_string(self, string):
        # infers the year and month from symbol, outputs a dictionary with keys contract_year and contract_month
        no_prefix_no_symbol = string.replace('F.US.' + self.symbol, '')
        contract_month = no_prefix_no_symbol[0]
        contract_year = no_prefix_no_symbol[1:]
        out_dict = dict()
        out_dict['contract_month'] = contract_month
        out_dict['contract_year'] = contract_year
        return out_dict

    def _infer_last_trading_date_in_expiry_month_for_contract(self, contract):
        # infers last exchange trading date for a given contracts expiry month
        # caution: this is not the roll date or the contracts last trading date !!!
        contract_expiry_info_dict = self._get_contract_year_and_month_string(contract)
        expiry_month = get_contract_month_from_code(contract_expiry_info_dict['contract_month'])
        proper_month, year_change = get_nth_previous_month(expiry_month, abs(self.roll_rule['roll_month']))

        if self.roll_rule.roll_type == 'standard':

            if year_change is False:
                last_exchange_trading_date_in_expiry_month = get_last_trading_day_of_month_for_exchange(
                    contract_expiry_info_dict['contract_year'], proper_month, exchange=self.exchange)
            else:
                wrong_year_str = contract_expiry_info_dict['contract_year']
                proper_year = get_nth_previous_year(wrong_year_str, 1)  # maximum shift is one year
                last_exchange_trading_date_in_expiry_month = get_last_trading_day_of_month_for_exchange(
                    proper_year, proper_month, exchange=self.exchange)

        else:
            assert self.roll_rule.roll_type == 'fixed_weekday'
            if year_change is False:
                last_exchange_trading_date_in_expiry_month = get_nth_weekday_of_contract_month(
                    contract_expiry_info_dict['contract_year'], proper_month, exchange=self.exchange,
                    nth_week=self.roll_rule.nth_week, nthweekday=self.roll_rule.fixed_week_day)
            else:
                wrong_year_str = contract_expiry_info_dict['contract_year']
                proper_year = get_nth_previous_year(wrong_year_str, 1)  # maximum shift is one year
                last_exchange_trading_date_in_expiry_month = get_nth_weekday_of_contract_month(
                    proper_year, proper_month, exchange=self.exchange,
                    nth_week=self.roll_rule.nth_week, nthweekday=self.roll_rule.fixed_week_day)

        return last_exchange_trading_date_in_expiry_month

    def _infer_roll_date_info_for_contract(self, contract):
        """
        :param contract: contract symbol
        :return dates_dict: dict of dates per contract
        """

        # gather required info from symbol and roll rule
        last_exchange_trading_date_in_expiry_month = self._infer_last_trading_date_in_expiry_month_for_contract(
            contract)
        roll_rule = self.roll_rule

        # get roll day from last bd in expiry month and subtract days according to roll rule
        last_trade_date = (last_exchange_trading_date_in_expiry_month - BDay(
            -roll_rule['last_relative_trading_day'])).date()

        # infer roll date from last trade date and roll rule
        roll_date = (last_trade_date - BDay(-roll_rule['roll_rule']))

        # exception handling: if inferred roll date not in index, find nearest prev date for which a price exists

        if pd.to_datetime(roll_date) not in self.dates:
            # corner case: when roll dates are too far into the future, set roll date to final date
            if roll_date <= self.dates[-1]:
                if roll_date > self.dates[0]:
                    roll_date = nearest(self.dates, roll_date)

        # corner cases for roll_out_dates
        if pd.to_datetime(roll_date) in self.dates:
            previous_bd_from_roll_date_loc = self.dates.get_loc(roll_date) - 1
            if previous_bd_from_roll_date_loc > 0:
                roll_out_date = self.dates[previous_bd_from_roll_date_loc]
            else:
                # case first day == roll_date (e.g. palladium contracts)
                roll_out_date = pd.NaT
        else:
            # case where roll date is in the future after last price date
            roll_out_date = roll_date - BDay(1)

        roll_info_contract = pd.Series(index=['roll_date', 'roll_out_date', 'last_trade_date'],
                                       data=[roll_date, roll_out_date, last_trade_date])
        roll_info_contract.name = contract

        return roll_info_contract

    def _create_roll_calendar(self):

        roll_dates_df = pd.DataFrame()

        for ticker in self.contracts:
            roll_info_contract = self._infer_roll_date_info_for_contract(ticker)
            if roll_info_contract.roll_date.date() < roll_info_contract.last_trade_date.date():
                # corner case: first contract for which roll date is prior to first data point to be neglected
                roll_dates_df = roll_dates_df.append(roll_info_contract)

        roll_dates_df['contract'] = roll_dates_df.index
        roll_dates_df = roll_dates_df.sort_values('roll_date')

        # cut off excess contracts in the future and in the past (i.e. where roll_dates are not applicable)
        # calendar_without_excess_contracts_future = roll_dates_df[roll_dates_df.roll_out_date <= self.dates[-1]].copy()
        calendar_without_excess_contracts = roll_dates_df[
            roll_dates_df.roll_out_date >= self.dates[0]].copy()

        # convert to pd.Timestamp
        return calendar_without_excess_contracts

    def _create_near_far_contract_table(self, cal, write_to_disk=False):
        """
        :param cal: calendar created from method above
        :param write_to_disk: boolean, if True writes to disk
        :return: returns a pd.Dataframe with index=price-dates, columns: near, far with respective symbols
        """
        # cut off irrelevant dates in contract calendar (either where roll date too early or too late)
        calendar_without_excess_contracts = cal[cal.roll_out_date <= self.dates[-1]].copy()

        roll_out_dates = calendar_without_excess_contracts.roll_out_date

        # get the contracts with the 2 closest roll_out dates after final price date (final near and final far)
        final_contracts = cal[cal.roll_out_date > self.dates[-1]].iloc[:2].contract.values

        # create near_far contract table for all valid roll out dates
        nf_df_at_roll_out = pd.DataFrame(index=roll_out_dates, columns=['near', 'far'])
        nf_df_at_roll_out['near'] = calendar_without_excess_contracts.contract.values
        nf_df_at_roll_out['far'] = calendar_without_excess_contracts.contract.shift(-1).values
        # the far contract at the last roll_out date is the near contract at the very last price date
        nf_df_at_roll_out['far'].iloc[-1] = final_contracts[0]

        # create near_far contract table for all price dates
        near_far_contracts_df = pd.DataFrame(index=self.dates, columns=['near', 'far'])
        near_far_contracts_df.iloc[-1] = final_contracts  # used to backfill end up to final rollout date
        near_far_contracts_df.loc[nf_df_at_roll_out.index] = nf_df_at_roll_out.values
        near_far_contracts_df.bfill(inplace=True)
        self.near_contracts = near_far_contracts_df.near
        self.far_contracts = near_far_contracts_df.far

        if write_to_disk:
            near_far_contracts_df.to_csv('C:/Users/28ide/Data/Futures/futures_data/switching_cycles/' + self.symbol +
                                         '_switching_cycle.csv')

        return near_far_contracts_df

    @staticmethod
    def create_previous_day_near_contract(near_contract):
        prev_day_near = near_contract.shift(1).copy()
        prev_day_near.iloc[0] = near_contract.iloc[0]  # first day has no prev day near, fill in with current near
        return prev_day_near

    def _create_near_far_series(self, near_far_contracts, expiry_calendar):

        # to calculate cont series I also need the value of the old near contract at the roll day
        old_near_contracts = near_far_contracts.near.shift(1).bfill()

        # create near far close
        near_far_close = pd.DataFrame()

        # entry to exit dates contract (normally roll to roll_out)
        first_entry_date = pd.Index([self.dates[0]])
        subsequent_entry_dates = pd.Index(expiry_calendar[expiry_calendar.roll_date < self.dates[-1]].roll_date.values)
        entry_dates = first_entry_date.append(subsequent_entry_dates)
        # todo: problem with entry dates, 2 times the same value:: wrong roll out date for PLEN95
        exit_dates = pd.Index(expiry_calendar[expiry_calendar.roll_out_date < self.dates[-1]].roll_out_date.values)
        last_exit_date = pd.Index([self.dates[-1]])
        exit_dates = exit_dates.append(last_exit_date)

        window = dict(zip(entry_dates, exit_dates))
        raw_data = self.raw_data.copy()

        for entry_date, exit_date in window.items():
            near_contract = near_far_contracts.loc[entry_date].near
            far_contract = near_far_contracts.loc[entry_date].far
            old_near_contract = old_near_contracts.loc[entry_date]
            data_slice = raw_data[entry_date:exit_date]

            close_slice_near = data_slice[data_slice.ticker == near_contract]['c']
            close_slice_near.name = 'near'

            close_slice_far = data_slice[data_slice.ticker == far_contract]['c']
            close_slice_far.name = 'far'
            close_slice_old_near = pd.Series(index=close_slice_near.index, data=np.nan)
            try:
                close_slice_old_near.loc[entry_date.date()] = data_slice[
                    data_slice.ticker == old_near_contract]['c'].loc[entry_date.date()]
            except Exception as e:
                # corner case where I do not have a data point for the old_near_contract at the roll_date
                # overwrite with new near (I lose any adjustment for this roll)
                # todo: what about the case where I roll too early? Ie there is no new near data point
                close_slice_old_near.loc[entry_date.date()] = close_slice_near.loc[entry_date.date()]
                print('{} exception with missing data for contract {}'.format(e, old_near_contract))
                # missing data case should be covered for by using only active contracts per market
            close_slice_old_near.name = 'old_near_at_roll'
            # todo: nan-handling
            df_window = pd.concat([close_slice_near, close_slice_far, close_slice_old_near], axis=1)
            df_window.near = df_window.near.astype(float)
            df_window.old_near_at_roll = df_window.old_near_at_roll.astype(float)
            df_window.loc[df_window['far'].isna(), 'far'] = df_window['near']

            near_far_close = pd.concat([near_far_close, df_window], axis=0)

        # ffill nas for near (example: ple no price for near contract at roll out day)
        near_far_close['near'] = near_far_close['near'].ffill()
        self.near_prices = near_far_close['near']
        self.far_prices = near_far_close['far']
        # self.near_contracts = near_far_close['near_contract']
        # self.far_contracts = near_far_close['far_contract']
        return near_far_close

    def _create_adjustment_series(self, near_far_close, cal):
        # calculate a series of cumulative adjustments to be subtracted from near price to get continuous series
        roll_dates = cal.roll_date[cal.roll_date < self.dates[-1]].copy().values
        roll_out_dates = cal.roll_out_date[cal.roll_out_date < self.dates[-1]].copy().values

        data_for_roll_adjustment = near_far_close[['near', 'old_near_at_roll']].loc[roll_dates].copy()

        # checked: return at roll_day is correct: old_contract_eod / old_contract_eod_minus_1
        adjustments = data_for_roll_adjustment['near'] - data_for_roll_adjustment['old_near_at_roll']

        adjustment_series = pd.Series(index=near_far_close.index, data=np.nan).fillna(0)

        adjustment_series.loc[roll_out_dates] = adjustments.values
        adjustment_series = adjustment_series[::-1].cumsum().sort_index()

        return adjustment_series

    @staticmethod
    def create_continuous_futures_series(near_far_close, adjustment_series):
        continuous_future = near_far_close.near + adjustment_series
        continuous_future.name = 'cont'
        return continuous_future

    def create_futures_price_df(self, write_to_disk=False):
        # create price_df with info: near_price, far_price, cont_price, 'near_contract', 'far_contract')
        expiry_calendar = self._create_roll_calendar()
        near_far_contracts = self._create_near_far_contract_table(expiry_calendar, write_to_disk=False)
        near_far_prices = self._create_near_far_series(near_far_contracts, expiry_calendar)
        adjustment_series = self._create_adjustment_series(near_far_prices, expiry_calendar)
        continuous_future = self.create_continuous_futures_series(near_far_prices, adjustment_series)

        futures_price_df = pd.DataFrame(index=near_far_prices.index, columns=['near', 'far', 'cont'])
        for column in futures_price_df:
            if column is 'cont':
                futures_price_df[column] = continuous_future
            else:
                futures_price_df[column] = near_far_prices[column].copy()

        if write_to_disk:
            futures_price_df.to_csv('C:/Users/28ide/Data/Futures/futures_data/continuous_futures/' + self.symbol +
                                    '_prices.csv')
            contract_table = pd.concat([self.near_contracts, self.far_contracts], axis=1)
            contract_table.to_csv('C:/Users/28ide/Data/Futures/futures_data/continuous_futures/' + self.symbol +
                                  '_contracts.csv')

        return futures_price_df

    def load_futures_data_from_disk(self):
        # data for full universe should then have a fct that calls this fct and writes data_df into a dataset with
        # key self.symbol
        path = self.data_path.parents[0].joinpath('continuous_futures').joinpath(self.symbol + '_prices.csv')
        if path.exists():
            data_df = pd.read_csv(path, index_col=0, parse_dates=True)
            print(" loaded data for market {} from disk".format(self.symbol))

        else:
            self.create_futures_price_df(write_to_disk=True)
            print("stored continuous futures data for market {} to disk".format(self.symbol))
            data_df = pd.read_csv(path, index_col=0, parse_dates=True)
        data_df.name = self.symbol
        return data_df

# tya.raw_data.loc[near.index[4000]][tya.raw_data.loc[near.index[4000]]['ticker']=='F.US.TYAM12']['c']

# -------------------------------------------


class FuturesData(object):

    def __init__(self, universe):
        self.markets = universe
        self.markets_data = DataSet()
        self.price_data = DataSet()
        self.returns = pd.DataFrame()
        self.futures = DataSet()

    def _read_futures_data(self):
        for market in self.markets:
            tmp = Future(market)
            self.futures[market] = tmp
            data = tmp.load_futures_data_from_disk()
            self.markets_data[market] = data

    def _get_cross_sectional_view(self, attribute='cont'):
        df = pd.DataFrame()
        if bool(self.markets_data) is False:
            self._read_futures_data()
        for key in self.markets_data:
            data = self.markets_data[key][attribute]
            data.name = self.markets_data[key].name
            df = pd.concat([df, data], axis=1)
            df.name = attribute
        return df

    def get_futures_prices_dataset(self):
        series_types = ['cont', 'near', 'far']
        prices_dataset = DataSet()
        for series in series_types:
            prices_dataset[series] = self._get_cross_sectional_view(attribute=series)
        self.price_data = prices_dataset
        return prices_dataset

    def get_futures_returns(self):
        if self.returns.empty:
            returns = pd.DataFrame()
            if bool(self.price_data) is False:
                self.get_futures_prices_dataset()
            returns = self.price_data.cont.diff() / self.price_data.near.shift(1)
        else:
            returns = self.returns
        return returns