import numpy as np
import pandas as pd
from hurst import compute_Hc
from tqdm import tqdm

from config import all_coins, statistics_dir, timeframes
from data.csv_data import get_data


def bootstrap_Hc(series, num_samples=1000):
    """
    Bootstrap a confidence interval for the Hurst exponent of time series.

    Parameters:
        series (array-like): The time series data.
        num_samples (int): The number of bootstrap samples to create.

    Returns:
        lower_H (float): Lower bound of the Hurst exponent.
        upper_H (float): Upper bound of the Hurst exponent.
    """
    n = len(series)
    H_samples = []

    for _ in range(num_samples):
        sample = np.random.choice(series, size=n, replace=True)
        H, _, _ = compute_Hc(sample, kind="change", simplified=False)
        H_samples.append(H)

    lower_H = np.percentile(H_samples, 5)
    upper_H = np.percentile(H_samples, 95)

    return lower_H, upper_H


def calc_hurst(
    data_type: str = "log returns", to_excel: bool = False, to_csv: bool = True
):
    """
    Calculates the Hurst exponent for the data and saves it to an Excel file.
    """

    file_name = f"{statistics_dir}/hurst_{data_type.replace(' ', '_')}"

    results = pd.DataFrame()

    kind = "change"
    if data_type == "close":
        kind = "price"

    if data_type == "scaled":
        data_type = "log returns"

    for coin in tqdm(all_coins):
        for time in timeframes:
            Hs = []
            for df in get_data(coin, time, data_type):
                prices = df[data_type].values.tolist()
                Hs.append(compute_Hc(prices, kind=kind, simplified=False)[0])

            H = np.mean(Hs)

            if 0.45 < H < 0.55:
                hurst_result = "Brownian motion"
            elif H < 0.45:
                hurst_result = "Negatively correlated"
            elif H > 0.55:
                hurst_result = "Positively correlated"

            info = {
                "Coin": coin,
                "Time Frame": time,
                "Hurst exponent": H,
                "Result": hurst_result,
            }

            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    if to_excel:
        results.to_excel(f"{file_name}.xlsx", index=False)

    if to_csv:
        results.to_csv(f"{file_name}.csv", index=False)

    print(results["Result"].value_counts())
