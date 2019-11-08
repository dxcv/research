import math

import numpy as np
import pandas as pd
from scipy import linalg as la


def normalize_data_mat(data):
    """ scales the matrix """
    t, n = data.shape
    normed_data = data.apply(lambda x: (x - x.mean()) / x.std())
    return normed_data, t, n


def corr_pca(normed_data):
    """ pca decomposition of covariance matrix of normed matrix """

    cov_mat = np.cov(normed_data, rowvar=False, bias=False)
    eig_values, eig_vectors = la.eigh(np.nan_to_num(cov_mat))

    # sort from largest to lowest
    idx = eig_values[::-1].argsort()

    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]

    n = len(eig_values)

    # create the eigenvalue matrix
    eig_val_mat = np.eye(n) * eig_values

    assert np.allclose(np.array(cov_mat),np.array(eig_vectors).dot(np.array(eig_val_mat)).dot(np.array(eig_vectors.T)))

    return np.array(eig_vectors), eig_val_mat


def eig_cleaning(eig_vecs, eig_val_mat, N, T):
    """ clean eigen values by applying RMT """

    # calculate cut-off lambda
    lambda_max = 1 + N/T + 2 * math.sqrt(N/T)

    # how many EVs to include
    idx = np.sum(eig_val_mat >= lambda_max)

    # re-combined to get back cleaned correlation matrix
    cleaned_corr_mat = eig_vecs[:, :idx].dot(eig_val_mat[:idx, :idx] ).dot(eig_vecs[:, :idx].T)

    # force diagonal of corrmat to equal one
    np.fill_diagonal(cleaned_corr_mat,1)

    return cleaned_corr_mat


def calculate_clean_correlation_matrix(data):
    """
    :param data: df of returns
    :return: df: returns a eigen-cleaned correlation matrix
    """

    normed_data, T, N = normalize_data_mat(data)

    eig_vec_mat, eig_val_mat = corr_pca(normed_data)
    cleaned_correl_mat = eig_cleaning(eig_vec_mat, eig_val_mat, N, T)

    df = pd.DataFrame(data=cleaned_correl_mat, columns=data.columns, index=data.columns)

    return df


def calculate_exp_vol(series, min_periods=0):
    # calculates exponential moving vol

    window = series.shape[0]
    span = 2 * window - 1

    assert len(series) > min_periods, "Not enough data to calculate exponential vol, " \
                                      "min period %d > series length" % min_periods

    exp_vol = (series.ewm(span=span, adjust=False)).std(bias=False)[-1]

    return exp_vol


def calculate_cleaned_cov_mat(df):
    corr = calculate_clean_correlation_matrix(df)
    vars = df.apply(lambda x: calculate_exp_vol(x) ** 2)
    cov_mat = pd.DataFrame(index=corr.index, columns=corr.columns)
    cov_mat.values[[np.arange(cov_mat.shape[0])] * 2] = vars
    for j in range(corr.shape[1]):
        for i in range(corr.shape[0]):
            if j != i:
                cov_mat.iloc[i, j] = corr.iloc[i, j] * cov_mat.iloc[i, i] **(1/2) * cov_mat.iloc[j, j] ** (1/2)
    return cov_mat

