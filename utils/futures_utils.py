from trading_calendars import get_calendar
from pandas.tseries.offsets import BMonthEnd


def get_contract_month_from_code(month):
    # returns a future's contract month as a num
    codes = dict()
    codes['F'] = 1
    codes['G'] = 2
    codes['H'] = 3
    codes['J'] = 4
    codes['K'] = 5
    codes['M'] = 6
    codes['N'] = 7
    codes['Q'] = 8
    codes['U'] = 9
    codes['V'] = 10
    codes['X'] = 11
    codes['Z'] = 12
    return codes[month]

# get_calendar('CME').closes give you open and close for each day


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
    try:
        last_trading_day = trading_calendar.closes[trading_calendar.closes.apply(
            lambda x: (x.year == float(year) and x.month == float(month)))].iloc[-1].date()
        # todo: corner case where desired month, year combination is too far in the future!
    except:
        year = str(year)
        month = str(month)
        last_trading_day = BMonthEnd().rollback(year+'-' + month).date()
        print('trading day is too far into the future for CME calendar')
    return last_trading_day


def trading_day_check(date, exchange):
    # returns boolean True/False depending on whether supplied date is a trading day
    trading_calendar = get_calendar(exchange)
    tradings_days = trading_calendar.closes
    if date in tradings_days:
        trading_day = True
    else:
        trading_day = False
    return trading_day


def nearest(items, pivot):
    # helper fucntion returning nearest date if pivot date is not in items
    return min(items, key=lambda x: abs(x - pivot))
