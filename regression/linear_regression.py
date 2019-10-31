import statsmodels.api as sm


def linear_ls_regression(yx, constant=True):
    """
    :param yx: column 0 is y, column 1 is x
    :param constant:
    :return:
    """
    y = yx[yx.columns[0]].copy()
    x = yx[yx.columns[1]].copy()

    assert y.isna().sum() == 0, "y contains nas !!!"
    assert x.isna().sum() == 0, "x contains nas !!!"

    if constant:
        x = sm.add_constant(x)

    model = sm.OLS(y, x)
    results = model.fit()
    return results
