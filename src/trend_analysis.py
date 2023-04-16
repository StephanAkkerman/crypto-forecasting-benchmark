import pymannkendall as mk
import numpy as np
import pandas as pd

# Local imports
from data import all_coins, timeframes, read_csv

# Define a function to calculate the Sen's slope estimator
def sen_slope(x):
    n = len(x)
    if n == 1:
        return np.nan
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    return np.median(slopes)


def mod_mk(test, preprocess):
    # Test options are:
    # mk.hamed_rao_modification_test
    # mk.yue_wang_modification_test
    # mk.pre_whitening_modification_test
    # mk.trend_free_pre_whitening_modification_test

    mk_df = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            if preprocess:
                df = read_csv(coin, time, ["log returns"])
            else:
                df = read_csv(coin, time, ["close"])

            # https://pypi.org/project/pymannkendall/
            (
                trend,
                h,
                p,
                z,
                Tau,
                s,
                var_s,
                slope,
                intercept,
            ) = test(df)

            info = {
                "Coin": coin,
                "Time": time,
                "Trend": trend,
            }

            mk_df = pd.concat(
                [mk_df, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    print("Test:", test)
    print("No trend", len(mk_df[mk_df["Trend"] == "no trend"]))
    print("decreasing", len(mk_df[mk_df["Trend"] == "decreasing"]))
    print("increasing", len(mk_df[mk_df["Trend"] == "increasing"]))
    print(len(mk_df))


def all_tests():
    for test in [
        mk.hamed_rao_modification_test,
        mk.yue_wang_modification_test,
        mk.pre_whitening_modification_test,
        mk.trend_free_pre_whitening_modification_test,
    ]:
        mod_mk(test, True)


all_tests()
