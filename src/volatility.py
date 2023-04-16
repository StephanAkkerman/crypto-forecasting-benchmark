import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

from data import all_coins, timeframes, read_csv


def window_analysis():
    # for coin in all_coins:
    #    for time in timeframes:
    df = read_csv("BTC", "1m")
    df = np.log(df).diff().dropna()

    window = 90
    df["Long Window (90)"] = df["close"].rolling(window=window).std() * np.sqrt(window)
    window = 30
    df["Medium Window (30)"] = df["close"].rolling(window=window).std() * np.sqrt(
        window
    )
    window = 10
    df["Short Window (10)"] = df["close"].rolling(window=window).std() * np.sqrt(window)

    df["BTC Log Returns"] = df["close"]
    df[
        [
            "BTC Log Returns",
            "Long Window (90)",
            "Medium Window (30)",
            "Short Window (10)",
        ]
    ].plot(subplots=True, figsize=(8, 6))

    plt.show()


# window_analysis()

def show_total_vol():
    total = pd.read_csv(f"data/TOTAL/TOTAL_1d_vol.csv", index_col="date")
    coin = pd.read_csv(f"data/DOGE/DOGEUSDT_1d_vol.csv", index_col="date")

    # Show both volatilitys on the same graph
    total["close"].plot(figsize=(8, 6), label="Total Market Cap")
    coin["close"].plot(label="BTC")
    plt.show()


def percentiles():
    total = pd.read_csv(f"data/TOTAL/TOTAL_1d_vol.csv")
    coin = "BTC"
    coin = pd.read_csv(f"data/{coin}/{coin}USDT_1d_vol.csv")

    total_percentile = total.rank(pct=True)
    coin_percentile = coin.rank(pct=True)

    # difference
    diff = total_percentile - coin_percentile
    # print(len(diff[diff["volatility"] > 0]))

    # Show both volatilitys on the same graph
    diff["volatility"].plot(figsize=(8, 6), label="Total Market Cap")
    coin["volatility"].plot(label="BTC")
    total["volatility"].plot(label="Total Market Cap")
    plt.show()


def vol_diff(selected_coin: str, timeframe: str):
    total = pd.read_csv(f"data/TOTAL/TOTAL_{timeframe}_vol.csv")
    coin = pd.read_csv(f"data/{selected_coin}/{selected_coin}USDT_{timeframe}_vol.csv")

    # Subtract the TOTAL index volatility from the cryptocurrency volatility
    volatility_differences = coin["volatility"] - total["volatility"]

    # Set a threshold to decide if the difference is significant enough to consider
    # the volatilities as equal.
    threshold = 0.01

    # Identify periods of higher, equal, and lower volatility
    higher_volatility = volatility_differences > threshold
    equal_volatility = np.abs(volatility_differences) <= threshold
    lower_volatility = volatility_differences < -threshold

    # Create a DataFrame to store the periods and corresponding labels
    periods_df = pd.DataFrame(index=coin.index, columns=["Label"])
    periods_df.loc[higher_volatility, "Label"] = 2
    periods_df.loc[equal_volatility, "Label"] = 1
    periods_df.loc[lower_volatility, "Label"] = 0

    periods_df["volatility"] = coin["volatility"]

    # Remove nan
    periods_df = periods_df.dropna()

    # Create the plot
    cmap = ListedColormap(["r", "b", "g"])
    norm = BoundaryNorm(range(3 + 1), cmap.N)
    points = np.array([periods_df.index, periods_df["volatility"]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    labels = periods_df["Label"].astype(int)
    lc.set_array(labels)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # Set the indices
    total = total.set_index("date")
    total["volatility"].plot(figsize=(8, 6), label="TOTAL index", ax=ax1, legend=True, xlabel="")
    coin = coin.set_index("date")
    coin["volatility"].plot(ax=ax1, label=selected_coin, legend=True, xlabel="")

    # ax2.add_collection(lc)
    # Add index
    volatility_differences = volatility_differences.set_axis(coin.index)
    volatility_differences.plot(ax=ax2, label="Volatility Difference")
    plt.axhline(y=0, color="grey", linestyle="-", alpha=0.7)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    # plt.xlim(periods_df.index.min(), periods_df.index.max())
    # plt.ylim(-1.1, 1.1)
    # plt.show()

    # Plot the volatility differences

    # volatility_differences.plot(label="Volatility Difference")

    plt.show()

#vol_diff("BTC", "1m")
# vol_categories()
