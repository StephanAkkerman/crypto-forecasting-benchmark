import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

# Local imports
from vars import all_coins, timeframes
from csv_data import read_csv


def ETS_decomposition(crypto: str, timeframe: str, plot_res: bool = False):
    """
    Make a ETS decomposition of the data, using both additive and multiplcative models

    Parameters
    ----------
    crypto : str
        The name of the cryptocurrency
    timeframe : str
        The time frame of the data
    plot_res : bool, optional
        Plots the residuals of the models, by default False
    """

    # Also test other models, like "additive" or "multiplicative"
    # For timeframes smaller than 4h set the period

    df = read_csv(crypto, timeframe)

    additive = seasonal_decompose(df, model="additive")
    multiplative = seasonal_decompose(df, model="multiplicative")

    add_res = additive.resid.dropna()
    mult_res = multiplative.resid.dropna()

    if plot_res:
        # Show both plots together
        plt.rcParams.update({"figure.figsize": (6, 6)})
        multiplative.plot().suptitle("Multiplicative Decomposition", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        additive.plot().suptitle("Additive Decomposition", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:

        # Show both plots together
        _, axs = plt.subplots(2, 1, figsize=(8, 8))

        plot_acf(add_res, ax=axs[0])
        plot_acf(mult_res, ax=axs[1])

    plt.show()


def stl_trend_seasonality_strength(log_returns : bool = False):

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

    # Save to csv
    # stl_df.to_csv("data/stl_seasonality.csv", index=False)
    if log_returns:
        stl_df.to_excel("data/tests/stl_seasonality_log_returns.xlsx", index=False)
    else:
        stl_df.to_excel("data/tests/stl_seasonality.xlsx", index=False)

    # print(stl_df["seasonal_strength"] > 0.5)


def analyze_seasonality():
    stl_df = pd.read_csv("data/tests/stl_seasonality.csv")

    # Sort the df on the seasonal strength
    print(stl_df.sort_values(by="seasonal_strength", ascending=False))


if __name__ == "__main__":
    stl_trend_seasonality_strength(True)
    # analyze_seasonality()
