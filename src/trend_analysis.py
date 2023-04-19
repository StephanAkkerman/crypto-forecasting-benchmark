import pymannkendall as mk
import numpy as np
import pandas as pd

# Local imports
from data import all_coins, timeframes, read_csv


def trend_test(test, log_returns=False):
    # Test options are:
    # mk.hamed_rao_modification_test
    # mk.yue_wang_modification_test
    # mk.pre_whitening_modification_test
    # mk.trend_free_pre_whitening_modification_test

    results = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            if log_returns:
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

            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    print("Test:", test)
    print("No trend", len(results[results["Trend"] == "no trend"]))
    print("decreasing", len(results[results["Trend"] == "decreasing"]))
    print("increasing", len(results[results["Trend"] == "increasing"]))
    
    return results


def all_tests(log_returns):

    all_results = pd.DataFrame()

    for test in [
        mk.hamed_rao_modification_test,
        mk.yue_wang_modification_test,
        mk.pre_whitening_modification_test,
        mk.trend_free_pre_whitening_modification_test,
    ]:
        results = trend_test(test, log_returns)

        # Add test name as column
        if test == mk.hamed_rao_modification_test:
            results["Test"] = "Hamed Rao"
        elif test == mk.yue_wang_modification_test:
            results["Test"] = "Yue Wang"
        elif test == mk.pre_whitening_modification_test:
            results["Test"] = "Pre-whitening"
        elif test == mk.trend_free_pre_whitening_modification_test:
            results["Test"] = "Trend-free pre-whitening"

        # Merge results
        all_results = pd.concat([all_results, results], axis=0, ignore_index=True)

    # Save as .csv and .xlsx
    all_results.to_csv("data/tests/trend_results.csv", index=False)
    all_results.to_excel("data/tests/trend_results.xlsx", index=False)

all_tests(True)
