import pandas as pd


def find_universe_tickers(df_prices):
    """ function to get markets in universe as they expand """
    first_valid_indices = df_prices.apply(lambda x: x.first_valid_index())
    is_live_market = pd.DataFrame(index=df_prices.index, columns=df_prices.columns)
    is_live_market = is_live_market.apply(lambda x: x.name >= first_valid_indices, axis=1)

    return is_live_market
