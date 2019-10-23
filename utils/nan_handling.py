import pandas as pd
import numpy as np


def find_number_of_nans(df, axis=1):
    idx_all_nas = df.loc[df.isna().sum(axis=axis) == df.shape[axis]]
    idx_all_but_one_nas = df.loc[df.isna().sum(axis=1) == df.shape[axis] - 1]


def find_first_valid_date(df):
    date = df.first_valid_index()
    return date


def find_first_valid_date_where_n_larger_one(df):
    first_valid_dates = df.apply(lambda x: x.first_valid_index(), axis=0)
    valid_date = first_valid_dates.sort_values()[1]
    return valid_date
