import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from tqdm import tqdm

# Local imports
from config import all_coins, statistics_dir, timeframes
from data.csv_data import get_data


def seasonal_strength_test(
    data_type: str = "log returns",
    to_excel: bool = False,
    to_csv: bool = True,
    use_one_freq: bool = False,
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
    file_name = f"{statistics_dir}/stl_seasonality_{data_type.replace(' ', '_')}"

    for coin in tqdm(all_coins):
        for time in timeframes:
            # Save the data and make it a plot
            if time == "1m":
                # 1 hour, 1 day
                freqs = [60 * 2] if use_one_freq else [60, 60 * 2]
            if time == "15m":
                # 1 hour, 1 day
                freqs = [4 * 24] if use_one_freq else [4, 4 * 24]
            elif time == "4h":
                # 1 day, 1 week, 1 month
                freqs = [6 * 30] if use_one_freq else [6, 6 * 7, 6 * 30]
            elif time == "1d":
                # 1 week, 1 month, 1 quarter, 1 year
                freqs = [365] if use_one_freq else [7, 30, 90, 365]

            # Try all frequencies
            for freq in freqs:
                dfs = get_data(coin, time, data_type)
                seasonal_strengths = []
                for df in dfs:
                    # Perform STL decomposition
                    stl = STL(df.squeeze(), period=freq)
                    result = stl.fit()

                    # https://otexts.com/fpp3/stlfeatures.html
                    seasonal_strengths.append(
                        1
                        - np.var(result.resid) / np.var(result.seasonal + result.resid)
                    )

                # Calculate the average seasonal strength
                seasonal_strength = np.mean(seasonal_strengths)

                info = {
                    "Coin": coin,
                    "Time Frame": time,
                    "Frequency": freq,
                    "Seasonal Strength": seasonal_strength,
                }

                # Add to df
                stl_df = pd.concat(
                    [stl_df, pd.DataFrame(info, index=[0])],
                    axis=0,
                    ignore_index=True,
                )

    pivot_df = pd.pivot_table(
        stl_df,
        values="Seasonal Strength",
        index=["Time Frame", "Frequency"],
        aggfunc="mean",
    )

    print(pivot_df)

    # Save to Excel
    if to_excel:
        stl_df.to_excel(f"{file_name}.xlsx", index=False)

    if to_csv:
        stl_df.to_csv(f"{file_name}.csv", index=False)
