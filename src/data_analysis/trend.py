import pandas as pd
import pymannkendall as mk

# Local imports
from config import all_coins, timeframes, statistics_dir
from data.csv_data import read_csv


def trend_tests(log_returns: bool = True, as_csv: bool = True, as_excel: bool = False):
    """
    Performs four trend tests on the data and saves the results to an Excel file.

    Parameters
    ----------
    log_returns : bool, optional
        If True then uses the logarithmic returns, by default False
    """

    file_name = f"{statistics_dir}/trend_results"

    if log_returns:
        file_name = f"{file_name}_log_returns"

    all_results = pd.DataFrame()

    for test in [
        mk.hamed_rao_modification_test,
        mk.yue_wang_modification_test,
        mk.pre_whitening_modification_test,
        mk.trend_free_pre_whitening_modification_test,
    ]:
        results = trend_test(test, log_returns)

        if all_results.empty:
            all_results = pd.concat([all_results, results], axis=0, ignore_index=True)

        else:
            # Merge the DataFrames
            all_results = pd.merge(
                all_results,
                results,
                how="inner",
                on=["Coin", "Time Frame"],
            )

    # Save as .xlsx
    if as_excel:
        all_results.to_excel(f"{file_name}.xlsx", index=False)
    if as_csv:
        all_results.to_csv(f"{file_name}.csv", index=False)


def trend_test(test, log_returns=True, as_csv: bool = True) -> pd.DataFrame:
    """
    Performs a trend test on all coins and timeframes.

    Parameters
    ----------
    test : test function
        The trend test function to use.
    log_returns : bool, optional
        If True then use the logarithmic return data, by default True

    Returns
    -------
    pd.DataFrame
        The results of the trend test.
    """

    results = pd.DataFrame()

    if test == mk.hamed_rao_modification_test:
        test_name = "Hamed Rao"
    elif test == mk.yue_wang_modification_test:
        test_name = "Yue Wang"
    elif test == mk.pre_whitening_modification_test:
        test_name = "Pre-whitening"
    elif test == mk.trend_free_pre_whitening_modification_test:
        test_name = "Trend-free pre-whitening"

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
                "Time Frame": time,
                test_name: trend,
            }

            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    # print("Test:", test)
    # print("No trend", len(results[results["Trend"] == "no trend"]))
    # print("decreasing", len(results[results["Trend"] == "decreasing"]))
    # print("increasing", len(results[results["Trend"] == "increasing"]))

    return results
