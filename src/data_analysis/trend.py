import pandas as pd
import pymannkendall as mk

# Local imports
from data.vars import all_coins, timeframes
from data.csv_data import read_csv

def trend_tests(log_returns : bool = False):
    """
    Performs four trend tests on the data and saves the results to an Excel file.

    Parameters
    ----------
    log_returns : bool, optional
        If True then uses the logarithmic returns, by default False
    """

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

    # Save as .xlsx
    all_results.to_excel("data/tests/trend_results.xlsx", index=False)

def trend_test(test, log_returns=False) -> pd.DataFrame:
    """
    Performs a trend test on all coins and timeframes.

    Parameters
    ----------
    test : test function
        The trend test function to use.
    log_returns : bool, optional
        If True then use the logarithmic return data, by default False

    Returns
    -------
    pd.DataFrame
        The results of the trend test.
    """

    results = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            if log_returns:
                df = read_csv(coin, time, ["log returns"]).dropna()
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