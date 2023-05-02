import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# Local imports
from vars import all_coins, timeframes
from csv_data import read_csv


def window_analysis(coin, time):
    # for coin in all_coins:
    #    for time in timeframes:
    df = read_csv(coin, time, ["log returns"])

    window = 90
    df["Long Window (90)"] = df["log returns"].rolling(window=window).std() * np.sqrt(
        window
    )
    window = 30
    df["Medium Window (30)"] = df["log returns"].rolling(window=window).std() * np.sqrt(
        window
    )
    window = 10
    df["Short Window (10)"] = df["log returns"].rolling(window=window).std() * np.sqrt(
        window
    )

    df["BTC Log Returns"] = df["log returns"]
    df[
        [
            "BTC Log Returns",
            "Long Window (90)",
            "Medium Window (30)",
            "Short Window (10)",
        ]
    ].plot(subplots=True, figsize=(8, 6))

    plt.show()


def vol_diff(selected_coin: str, timeframe: str):
    total = read_csv("TOTAL", timeframe, ["volatility"])
    coin = read_csv(selected_coin, timeframe, ["volatility"])

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
    total["volatility"].plot(
        figsize=(8, 6), label="TOTAL index", ax=ax1, legend=True, xlabel=""
    )
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


def avg_vol(timeframe: str = "1d"):
    total_vol = 0

    for coin in all_coins:
        coin = read_csv(coin, timeframe, ["volatility"]).dropna()
        total_vol += coin["volatility"].mean()

    return total_vol / len(all_coins)


def plot_all_volatilies(timeframe="1d"):
    # Plot all volatilities
    complete_df = pd.DataFrame()

    for coin in all_coins:
        coin_df = read_csv(coin, timeframe, ["volatility"]).dropna()

        # Set the index to the dates from coin_df
        if complete_df.empty:
            complete_df.index = coin_df.index

        complete_df[coin] = coin_df["volatility"].tolist()

    ax = complete_df.plot(figsize=(12, 6), alpha=0.3, legend=False)

    # Calculate the average of all volatilities
    avg_volatility = complete_df.mean(axis=1)

    # Plot the average volatility as a big red line with increased width
    avg_line = plt.plot(
        avg_volatility, color="red", linewidth=2, label="Average Volatility"
    )

    # Calculate the overall average of the avg_volatility and plot it as a horizontal blue line
    overall_avg_volatility = avg_volatility.mean()
    overall_avg_line = plt.axhline(
        y=overall_avg_volatility,
        color="blue",
        linewidth=2,
        label="Overall Average Volatility",
    )

    # Show legends only for the average volatility and overall average volatility lines
    ax.legend(handles=[avg_line[0], overall_avg_line], loc="best")

    # Set y-axis title
    ax.set_ylabel('Volatility')
    ax.set_xlabel("Date")

    plt.show()


plot_all_volatilies("1m")
