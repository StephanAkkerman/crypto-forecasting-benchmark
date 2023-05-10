import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# Local imports
from data.vars import all_coins, timeframes
from data.csv_data import read_csv

def volatility_tests():
    """
    Main function to perform all volatility tests
    """
    window_analysis()
    vol_diff()
    plot_percentiles()
    percentiles_table()


def window_analysis(coin="BTC", time="1d"):
    """
    Shows the volatility of the data using different windows.

    Parameters
    ----------
    coin : str, optional
        The symbol of the cryptocurrency, by default "BTC"
    time : str, optional
        The time frame to be used, by default "1d"
    """
    
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
    axes = df[
        [
            "BTC Log Returns",
            "Long Window (90)",
            "Medium Window (30)",
            "Short Window (10)",
        ]
    ].plot(subplots=True, figsize=(12, 8))

    # Put all legends right upper
    for ax in axes:
        ax.legend(loc="upper right")
        ax.set_ylabel("Volatility")  # Set y-axis label for each subplot

    axes[0].set_ylabel("Log Returns")  # Set y-axis label for the first subplot

    # Change x-axis labels
    axes[-1].set_xlabel("Date")  # Only set x-axis label for the last subplot

    plt.show()
    plt.savefig("data/plots/window_analysis.png")


def vol_diff(selected_coin: str = "BTC", timeframe: str = "1d"):
    """
    Calculates the difference in volatility between the selected coin and the TOTAL index.

    Parameters
    ----------
    selected_coin : str, optional
        Symbol of the cryptocurrency, by default "BTC"
    timeframe : str, optional
        The time frame to be used, by default "1d"
    """
    
    total = pd.read_csv(f"data/TOTAL/TOTAL_{timeframe}.csv").dropna()
    coin = read_csv(selected_coin, timeframe, ["volatility"]).dropna()
    dates = total["date"]

    total.reset_index(inplace=True)
    coin.reset_index(inplace=True)

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
        figsize=(12, 8), label="TOTAL index", ax=ax1, legend=True, xlabel=""
    )
    # coin = coin.set_index("date")
    coin["volatility"].plot(ax=ax1, label=selected_coin, legend=True, xlabel="")

    # ax2.add_collection(lc)
    # Add index
    volatility_differences = volatility_differences.set_axis(dates)
    volatility_differences.plot(ax=ax2, label="Volatility Difference")
    plt.axhline(y=0, color="grey", linestyle="-", alpha=0.7)

    ax1.legend(loc="upper right")
    # ax2.legend(loc="upper right")

    # plt.xlim(periods_df.index.min(), periods_df.index.max())
    # plt.ylim(-1.1, 1.1)
    # plt.show()

    # Plot the volatility differences

    # volatility_differences.plot(label="Volatility Difference")

    ax1.set_ylabel("Volatility")
    ax2.set_ylabel("Volatility Difference")

    ax1.set_xlabel("")
    ax2.set_xlabel("Date")

    plt.show()
    plt.savefig("data/plots/volatility_difference.png")

def avg_vol(timeframe: str = "1d") -> float:
    """
    Calculates the average volatility of all cryptocurrencies.

    Parameters
    ----------
    timeframe : str, optional
        The time frame to use, by default "1d"

    Returns
    -------
    float
        The average volatility of all cryptocurrencies.
    """
    
    total_vol = 0

    for coin in all_coins:
        coin = read_csv(coin, timeframe, ["volatility"]).dropna()
        total_vol += coin["volatility"].mean()

    return total_vol / len(all_coins)


def plot_all_volatilies(timeframe="1d"):
    """
    Plots the volatility of all cryptocurrencies and the average volatility.

    Parameters
    ----------
    timeframe : str, optional
        The time frame to use, by default "1d"
    """

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
    ax.set_ylabel("Volatility")
    ax.set_xlabel("Date")

    plt.show()
    plt.savefig("data/plots/avg_volatility.png")

def plot_percentiles(timeframe="1d"):
    """
    Plots all volatilities and the 25th, 50th, and 75th percentile.

    Parameters
    ----------
    timeframe : str, optional
        The time frame to use for the data, by default "1d"
    """
    
    complete_df = pd.DataFrame()

    for coin in all_coins:
        coin_df = read_csv(coin, timeframe, ["volatility"]).dropna()

        # Set the index to the dates from coin_df
        if complete_df.empty:
            complete_df.index = coin_df.index

        complete_df[coin] = coin_df["volatility"].tolist()

    ax = complete_df.plot(figsize=(12, 6), alpha=0.3, legend=False)

    # Calculate the overall 50th percentile (median) of all volatilities
    overall_median_volatility = complete_df.stack().median()

    # Plot the overall median volatility as a horizontal blue line
    overall_median_line = plt.axhline(
        y=overall_median_volatility,
        color="blue",
        linewidth=2,
        label="Overall Median Volatility",
    )

    # Calculate the overall 75th percentile of all volatilities
    overall_q3_volatility = complete_df.stack().quantile(0.75)
    print(overall_q3_volatility)

    # Plot the overall 75th percentile volatility as a horizontal green line
    overall_q3_line = plt.axhline(
        y=overall_q3_volatility,
        color="lime",
        linewidth=2,
        label="Overall 75th Percentile Volatility",
    )

    overall_q1_volatility = complete_df.stack().quantile(0.25)
    print(overall_q1_volatility)

    # Plot the overall 25th percentile volatility as a horizontal blue line
    overall_q1_line = plt.axhline(
        y=overall_q1_volatility,
        color="red",
        linewidth=2,
        label="Overall 25th Percentile Volatility",
    )

    # Show legends only for the overall median volatility and overall 75th percentile volatility lines
    ax.legend(
        handles=[overall_median_line, overall_q3_line, overall_q1_line], loc="best"
    )

    # Set y-axis title
    ax.set_ylabel("Volatility")
    ax.set_xlabel("Date")

    plt.show()
    plt.savefig("data/plots/volatility_percentiles.png")

def percentiles_table():
    """
    Displays the 25th and 75th percentile for each time frame.
    """
    
    complete_df = pd.DataFrame()

    for timeframe in timeframes:
        for coin in all_coins:
            coin_df = read_csv(coin, timeframe, ["volatility"]).dropna()

            # Set the index to the dates from coin_df
            if complete_df.empty:
                complete_df.index = coin_df.index

            complete_df[coin] = coin_df["volatility"].tolist()
            
        print(f"25th percentile for {timeframe}: {complete_df.stack().quantile(0.25)}")
        print(f"75th percentile for {timeframe}: {complete_df.stack().quantile(0.75)}")
        print()