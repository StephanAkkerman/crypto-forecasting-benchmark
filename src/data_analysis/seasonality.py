import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.seasonal import STL

# Local imports
from config import all_coins, timeframes, statistics_dir
from data.csv_data import read_csv


def seasonal_strength_test(
    log_returns: bool = True, to_excel: bool = False, to_csv: bool = True
):
    """
    Calculates the strength of the trend and seasonality for the data using STL decomposition.
    Formula is based on https://otexts.com/fpp3/stlfeatures.html

    Parameters
    ----------
    log_returns : bool, optional
        If True then uses the logarithmic returns instead, by default False
    """

    stl_df = pd.DataFrame()
    file_name = f"{statistics_dir}/stl_seasonality"

    if log_returns:
        file_name = f"{file_name}_log_returns"

    for coin in tqdm(all_coins):
        for time in timeframes:
            if log_returns:
                df = read_csv(coin, time, col_names=["log returns"]).dropna()
            else:
                df = read_csv(coin, time, col_names=["close"]).dropna()

            # Save the data and make it a plot
            if time == "1m":
                # 1 hour, 1 day
                freqs = [60 * 2]  # [60, 60 * 2]
            if time == "15m":
                # 1 hour, 1 day
                freqs = [4 * 24]  # [4, 4 * 24]
            elif time == "4h":
                # 1 day, 1 week, 1 month
                freqs = [6 * 30]  # [6, 6 * 7, 6 * 30]
            elif time == "1d":
                # 1 week, 1 month, 1 quarter, 1 year
                freqs = [365]  # [7, 30, 90, 365]

            # Try all frequencies
            for freq in freqs:
                # Perform STL decomposition
                stl = STL(df.squeeze(), period=freq)
                result = stl.fit()

                # https://otexts.com/fpp3/stlfeatures.html
                seasonal_strength = 1 - np.var(result.resid) / np.var(
                    result.seasonal + result.resid
                )

                info = {
                    "Coin": coin,
                    "Time Frame": time,
                    # "Frequency": freq,
                    "Seasonal Strength": seasonal_strength,
                }

                # Add to df
                stl_df = pd.concat(
                    [stl_df, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
                )

    # Save to Excel
    if to_excel:
        stl_df.to_excel(f"{file_name}.xlsx", index=False)

    if to_csv:
        stl_df.to_csv(f"{file_name}.csv", index=False)
