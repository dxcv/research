from trading_calendars import get_calendar
import pandas as pd


def get_last_trading_day_of_month_for_exchange(year, month, exchange):
    " eg. 2013, 04, CME"
    if type(month) == str:
        if month.startswith('0'):
            month = month.lstrip('0')

    # covers cases for str/int input where year is provided only as XX or YY
    if type(year) == int or type(year) is float:
        if len(str(year)) == 2:
            if year >= 90:
                year = year + 1900
            else:
                year = year + 2000
        else:
            year = year

    else:
        assert type(year) == str
        if len(year) == 2:
            if int(year) >= 90:
                year = '19' + year
            else:
                year = '20' + year
        else:
            year = year

    trading_calendar = get_calendar(exchange)
    last_trading_day = trading_calendar.closes[trading_calendar.closes.apply(
        lambda x: (x.year == float(year) and x.month == float(month)))].iloc[-1].date()
    return last_trading_day


def get_nth_last_trading_day_of_year_month(year, month, day_from_last, exchange):
    # returns the nth last trading date for a specified year/month combination as a pandas timestamp
    if type(year) == float or type(year) == int:
        year = str(year)
    if type(month) == float or type(month) == int:
        month = str(month)
    trading_calendar = get_calendar(exchange)
    date = trading_calendar.closes[year + '-' + month].iloc[- day_from_last]
    return date


def get_nth_previous_month(month, n_previous):
    year_change = False
    if n_previous != 0:
        if month // n_previous <= 0:
            difference = month - n_previous
            proper_month = 12 + difference
            year_change = True
        elif month / n_previous == 1:
            proper_month = 12
            year_change = True
        else:
            proper_month = month - n_previous
    else:
        proper_month = month
    return proper_month, year_change


def get_nth_previous_year(year, n_previous):
    assert abs(float(n_previous)) <= 1
    if type(year) == str and len(year) == 2:
        if float(year) >= 90:
            year = '19' + year
        else:
            year = '20' + year
    else:
        assert type(year) == float or type(year) == int
        year = str(year)
        if len(year) == 2:
            if float(year) >= 90:
                year = '19' + year
            else:
                year = '20' + year
    year_num = float(year)
    if type(n_previous) == str:
        n_previous = float(n_previous)
    proper_year = year_num - n_previous

    return int(proper_year)


def get_nth_weekday_of_contract_month(year, month, exchange, nth_week, nthweekday):
    """
    :param year: year as string or float
    :param month: month as sring or float
    :param exchange: relevant exchange calendar
    :param nth_week: which week in month
    :param nthweekday: which weekday, Mo to Fr weekdays are from 0 to 6
    :return: eg get third friday of trading month nth_week = 3, nthweekday = 4
    """
    if type(month) == float or type(month) == int:
        month = str(month)

    if type(year) == int or type(year) is float:
        if len(str(year)) == 2:
            if year >= 90:
                year = year + 1900
            else:
                year = year + 2000
        else:
            year = year

    else:
        assert type(year) == str
        if len(year) == 2:
            if int(year) >= 90:
                year = '19' + year
            else:
                year = '20' + year
        else:
            year = year

    trading_calendar = get_calendar(exchange)
    dates_in_expiry_month = trading_calendar.closes[year + '-' + month]

    if type(nthweekday) == str:
        nthweekday = float(nthweekday)

    week_number_in_year_entry = dates_in_expiry_month.dt.week.iloc[0]

    # check with standard iso calendar to see whether we need to adjust expiry day (e.g. gap in trading calendar)
    date_calendar = pd.date_range(start=year + '-' + month, periods=31)

    # base expiry date
    expiry_date = dates_in_expiry_month[dates_in_expiry_month.dt.weekday == nthweekday].iloc[nth_week - 1]

    # corner case when there is a nth weekday missing from trading calendar: roll one week before
    if date_calendar[date_calendar.weekday == nthweekday][nth_week-1].date() != expiry_date.date():
        print('Warning: expiry date {} rolled forward by 1 week'.format(expiry_date))
        expiry_date = dates_in_expiry_month[dates_in_expiry_month.dt.weekday == nthweekday].iloc[nth_week - 2]

    return expiry_date

