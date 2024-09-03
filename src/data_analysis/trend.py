from collections import Counter

import pandas as pd
import pymannkendall as mk
from tqdm import tqdm

# Local imports
from config import all_coins, statistics_dir, timeframes
from data.csv_data import get_data


def find_majority(row):
    # Count the frequency of each unique result in the row
    counter = Counter(row)
    # Find the most common result
    most_common_result, _ = counter.most_common(1)[0]
    return most_common_result


def trend_tests(
    data_type: str = "log returns",
    as_csv: bool = True,
    as_excel: bool = False,
    use_majority: bool = False,
):
    """
    Performs four trend tests on the data and saves the results to an Excel file.

    Parameters
    ----------
    log_returns : bool, optional
        If True then uses the logarithmic returns, by default False
    """

    file_name = f"{statistics_dir}/trend_results_{data_type.replace(' ', '_')}"

    df = pd.DataFrame()

    for test in tqdm(
        [
            mk.hamed_rao_modification_test,
            mk.yue_wang_modification_test,
            mk.pre_whitening_modification_test,
            mk.trend_free_pre_whitening_modification_test,
        ]
    ):
        results = trend_test(test, data_type)

        if df.empty:
            df = pd.concat([df, results], axis=0, ignore_index=True)

        else:
            # Merge the DataFrames
            df = pd.merge(
                df,
                results,
                how="inner",
                on=["Coin", "Time Frame"],
            )

    if use_majority:
        # Apply the function across the rows
        df["Result"] = df.apply(find_majority, axis=1)

        # Change Results to trend if its increasing or decreasing
        df["Result"] = df["Result"].str.replace("increasing", "trend")
        df["Result"] = df["Result"].str.replace("decreasing", "trend")

        print(df["Result"].value_counts())
    else:
        # Print Results of each test
        print(df["Hamed Rao"].value_counts())
        print(df["ESS"].value_counts())
        print(df["Pre-whitening"].value_counts())
        print(df["Trend-free"].value_counts())

    # Save as .xlsx
    if as_excel:
        df.to_excel(f"{file_name}.xlsx", index=False)
    if as_csv:
        df.to_csv(f"{file_name}.csv", index=False)


def trend_test(test, data_type: str) -> pd.DataFrame:
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

    # https://pypi.org/project/pymannkendall/
    if test == mk.hamed_rao_modification_test:
        test_name = "Hamed Rao"
    elif test == mk.yue_wang_modification_test:
        test_name = "ESS"
    elif test == mk.pre_whitening_modification_test:
        test_name = "Pre-whitening"
    elif test == mk.trend_free_pre_whitening_modification_test:
        test_name = "Trend-free"

    for coin in all_coins:
        for time in timeframes:
            trend_results = []
            for df in get_data(coin, time, data_type):
                trend_results.append(test(df)[0])

            trend_result = trend_results[0]
            if len(trend_results) > 1:
                string_counts = Counter(trend_results)
                trend_result = string_counts.most_common(1)[0][0]

            info = {
                "Coin": coin,
                "Time Frame": time,
                test_name: trend_result,
            }

            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    return results


def trend_analysis(data_type: str = "log returns"):
    file_name = f"{statistics_dir}/trend_results_{data_type.replace(' ', '_')}.csv"

    df = pd.read_csv(file_name)

    # Print Results of each test
    print(df["Hamed Rao"].value_counts())
    print(df["Pre-whitening"].value_counts())
    print(df["Trend-free"].value_counts())
    print(df["ESS"].value_counts())
