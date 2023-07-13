import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

# Local imports
from data.vars import all_coins, timeframes
from data.csv_data import read_csv


def seasonal_strength_test(log_returns : bool = False):
    """
    Calculates the strength of the trend and seasonality for the data using STL decomposition.
    Formula is based on https://otexts.com/fpp3/stlfeatures.html

    Parameters
    ----------
    log_returns : bool, optional
        If True then uses the logarithmic returns instead, by default False
    """

    stl_df = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time)

            if log_returns:
                df = np.log(df).diff().dropna()

            # Save the data and make it a plot
            if time == "1m":
                # 1 hour, 1 day
                freqs = [60, 60 * 2]
            if time == "15m":
                # 1 hour, 1 day
                freqs = [4, 4 * 24]
            elif time == "4h":
                # 1 day, 1 week, 1 month
                freqs = [6, 6 * 7, 6 * 30]
            elif time == "1d":
                # 1 week, 1 month, 1 quarter, 1 year
                freqs = [7, 30, 90, 365]

            # Try all frequencies
            for freq in freqs:
                # Perform STL decomposition
                stl = STL(df, period=freq)
                result = stl.fit()

                # https://otexts.com/fpp3/stlfeatures.html
                seasonal_strength = 1 - np.var(result.resid) / np.var(
                    result.seasonal + result.resid
                )

                info = {
                    "coin": coin,
                    "time": time,
                    "freq": freq,
                    "seasonal_strength": seasonal_strength,
                }

                # Add to df
                stl_df = pd.concat(
                    [stl_df, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
                )

    # Save to Excel
    if log_returns:
        stl_df.to_excel("data/tests/stl_seasonality_log_returns.xlsx", index=False)
    else:
        stl_df.to_excel("data/tests/stl_seasonality.xlsx", index=False)
