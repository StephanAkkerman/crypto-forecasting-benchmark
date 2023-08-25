import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import plotly.graph_objects as go

# Local imports
from config import (
    all_coins,
    timeframes,
    small_cap,
    mid_cap,
    large_cap,
    plots_dir,
    n_periods,
    test_percentage,
    val_percentage,
)
from data.csv_data import read_csv


def volatility_tests():
    """
    Main function to perform all volatility tests
    """
    window_analysis()
    vol_diff()
    plot_percentiles()
    get_percentiles(display=True)


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

    plt.savefig(f"{plots_dir}/window_analysis.png")
    plt.show()


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

    plt.savefig(f"{plots_dir}/volatility_difference.png")
    plt.show()


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

    volatility_df = get_all_volatility_data(timeframe=timeframe)

    ax = volatility_df.plot(figsize=(12, 6), alpha=0.3, legend=False)

    # Calculate the average of all volatilities
    avg_volatility = volatility_df.mean(axis=1)

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

    plt.savefig(f"{plots_dir}/avg_volatility.png")
    plt.show()


def plot_percentiles(timeframe="1d"):
    """
    Plots all volatilities and the 25th, 50th, and 75th percentile.

    Parameters
    ----------
    timeframe : str, optional
        The time frame to use for the data, by default "1d"
    """

    volatility_df = pd.DataFrame()

    for coin in all_coins:
        coin_df = read_csv(coin, timeframe, ["volatility"]).dropna()

        # Set the index to the dates from coin_df
        if volatility_df.empty:
            volatility_df.index = coin_df.index

        volatility_df[coin] = coin_df["volatility"].tolist()

    ax = volatility_df.plot(figsize=(12, 6), alpha=0.3, legend=False)

    # Calculate the overall 50th percentile (median) of all volatilities
    overall_median_volatility = volatility_df.stack().median()

    # Plot the overall median volatility as a horizontal blue line
    overall_median_line = plt.axhline(
        y=overall_median_volatility,
        color="blue",
        linewidth=2,
        label="Overall Median Volatility",
    )

    # Calculate the overall 75th percentile of all volatilities
    overall_q3_volatility = volatility_df.stack().quantile(0.75)
    print(overall_q3_volatility)

    # Plot the overall 75th percentile volatility as a horizontal green line
    overall_q3_line = plt.axhline(
        y=overall_q3_volatility,
        color="lime",
        linewidth=2,
        label="Overall 75th Percentile Volatility",
    )

    overall_q1_volatility = volatility_df.stack().quantile(0.25)
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

    plt.savefig(f"{plots_dir}/volatility_percentiles.png")
    plt.show()


def get_percentile(volatility_df) -> (float, float):
    quantile25 = volatility_df.stack().quantile(0.25)
    quantile75 = volatility_df.stack().quantile(0.75)

    return quantile25, quantile75


def get_tf_percentile(time_frame: str) -> (float, float):
    return get_percentile(get_all_volatility_data(timeframe=time_frame))


def get_percentiles():
    """
    Displays the 25th and 75th percentile for each time frame.
    """

    for timeframe in timeframes:
        quantile25, quantile75 = get_percentile(
            get_all_volatility_data(timeframe=timeframe)
        )

        print(f"25th percentile for {timeframe}: {quantile25}")
        print(f"75th percentile for {timeframe}: {quantile75}")
        print()


def get_volatility(coin, time_frame):
    # Get the volatility
    coin_df = read_csv(
        coin=coin, timeframe=time_frame, col_names=["volatility"]
    ).dropna()

    # Convert to dataframe
    return pd.DataFrame(
        data=coin_df["volatility"].tolist(), columns=[coin], index=coin_df.index
    )


def get_all_volatility_data(timeframe: str = "1d") -> pd.DataFrame:
    """
    Gets the volatility data for all coins in the given timeframe.

    Parameters
    ----------
    timeframe : str
        The timeframe to use, by default "1d"

    Returns
    -------
    pd.DataFrame
        The volatility data for all coins in the given timeframe
    """

    # Save the data in this dataframe
    volatility_df = pd.DataFrame()

    for coin in all_coins:
        volatility_df = pd.concat(
            [volatility_df, get_volatility(coin, timeframe)], axis=1
        )

    # Identify columns with NaN at the first index
    cols_with_nan_at_first = volatility_df.columns[volatility_df.iloc[0].isna()]

    # Shift up the columns with NaN at the first index
    volatility_df[cols_with_nan_at_first] = volatility_df[cols_with_nan_at_first].shift(
        -1
    )

    # Drop the last row
    volatility_df = volatility_df.iloc[:-1]

    return volatility_df


def calculate_percentiles(volatility_df: pd.DataFrame):
    """
    Calculates the median, 0.75, and 0.25 percentiles for the given dataframe.

    Parameters
    ----------
    volatility_df : pd.DataFrame
        The dataframe to calculate the percentiles for

    Returns
    -------
    float, float, float
        median, 0.75, and 0.25 percentiles
    """

    overall_median_volatility = volatility_df.stack().median()
    overall_q3_volatility = volatility_df.stack().quantile(0.75)
    overall_q1_volatility = volatility_df.stack().quantile(0.25)

    return overall_median_volatility, overall_q3_volatility, overall_q1_volatility


def plot_lines(volatility_df: pd.DataFrame):
    """
    Plots the average volatility, median volatility, and 0.75 and 0.25 percentiles.

    Parameters
    ----------
    volatility_df : pd.DataFrame
        The dataframe to plot the lines for

    Returns
    -------
    matplotlib.lines.Line2D, matplotlib.lines.Line2D, matplotlib.lines.Line2D, matplotlib.lines.Line2D
        The lines for the average volatility, median volatility, 0.75 percentile, and 0.25 percentile
    """

    avg_volatility = volatility_df.mean(axis=1)
    avg_line = plt.plot(
        avg_volatility,
        color="dodgerblue",
        linewidth=2.5,
        alpha=0.7,
        label="Average Volatility",
    )

    (
        overall_median_volatility,
        overall_q3_volatility,
        overall_q1_volatility,
    ) = calculate_percentiles(volatility_df)

    overall_median_line = plt.axhline(
        y=overall_median_volatility,
        color="lime",
        linewidth=2,
        alpha=0.7,
        label="Overall Median Volatility",
    )
    overall_q3_line = plt.axhline(
        y=overall_q3_volatility,
        color="orange",
        linewidth=2,
        alpha=0.7,
        label="Overall 75th Percentile Volatility",
    )
    overall_q1_line = plt.axhline(
        y=overall_q1_volatility,
        color="darkred",
        linewidth=2,
        alpha=0.7,
        label="Overall 25th Percentile Volatility",
    )

    return avg_line, overall_median_line, overall_q3_line, overall_q1_line


def plot_train_test_periods(volatility_df: pd.DataFrame, ax: plt.Axes):
    """
    Plots the training, validation, and testing periods on the graph.

    Parameters
    ----------
    volatility_df : pd.DataFrame
        The dataframe containing the volatility data for all coins.
    ax : plt.Axes
        The axes to plot the lines on.

    Returns
    -------
    list, list, list
        The lines for the training, validation, and testing periods.
    """

    ts_length = 999
    test_size = int(ts_length / (1 / test_percentage - 1 + n_periods))
    train_size = int(test_size * (1 / test_percentage - 1))
    val_size = int(val_percentage * train_size)
    train_size = train_size - val_size

    _, ymax = ax.get_ylim()

    line_start = ymax * 2
    training_lines = []
    validation_lines = []
    testing_lines = []
    for i in range(n_periods):
        train_start = i * test_size
        train_end = train_start + train_size

        date_min = volatility_df.index.min()
        date_max = volatility_df.index.max()

        train_line_start = (volatility_df.index[train_start] - date_min) / (
            date_max - date_min
        )
        train_line_end = (volatility_df.index[train_end] - date_min) / (
            date_max - date_min
        )
        val_end = (volatility_df.index[train_end + val_size] - date_min) / (
            date_max - date_min
        )
        test_end = (volatility_df.index[min(train_end + test_size, 969)] - date_min) / (
            date_max - date_min
        )

        training_lines.append(
            plt.axhline(
                y=line_start,
                xmin=train_line_start,
                xmax=train_line_end,
                color="blue",
                linewidth=4,
                label="Training Periods",
            )
        )
        validation_lines.append(
            plt.axhline(
                y=line_start,
                xmin=train_line_end,
                xmax=val_end,
                color="green",
                linewidth=4,
                label="Validation Periods",
            )
        )
        testing_lines.append(
            plt.axhline(
                y=line_start,
                xmin=val_end,
                xmax=test_end,
                color="red",
                linewidth=4,
                label="Test Periods",
            )
        )
        line_start -= ymax * 0.1

    return training_lines, validation_lines, testing_lines


def plot_periods(timeframe="1d"):
    """
    Plots the number of periods and the training, validation, and testing periods.

    Parameters
    ----------
    timeframe : str, optional
        The time frame of the data, by default "1d"
    """

    # Get the volatility data
    volatility_df = get_all_volatility_data(timeframe)

    # Plot the volatility data
    ax = volatility_df.plot(figsize=(12, 6), alpha=0.2, color="grey", legend=False)

    # Get the lines for the average volatility, median volatility, and 0.75 and 0.25 percentiles
    avg_line, overall_median_line, overall_q3_line, overall_q1_line = plot_lines(
        volatility_df
    )

    # Get the lines for the training, validation, and testing periods
    training_lines, validation_lines, testing_lines = plot_train_test_periods(
        volatility_df, ax
    )

    # Create first legend
    first_legend = ax.legend(
        handles=[avg_line[0], overall_median_line, overall_q3_line, overall_q1_line],
        loc="best",
    )

    # Add the first legend manually to the current Axes.
    ax.add_artist(first_legend)

    # Create second legend
    ax.legend(
        handles=[training_lines[0], validation_lines[0], testing_lines[0]],
        loc="upper center",
        ncols=3,
        bbox_to_anchor=(0.5, 1.05),
    )

    ax.set_ylabel("Volatility")
    ax.set_xlabel("Date")

    plt.show()


def plotly_volatility(time_frame="1d", percentile_per_group=False):
    """
    Plot the volatility of all the coins in the dataset.
    Also plots the 25th and 75th percentiles of the volatility.

    Parameters
    ----------
    time_frame : str, optional
        The time frame to use, by default "1d"
    percentile_per_group : bool, optional
        If the percentile should be calculated by market cap group, by default False
    """

    vol_df = get_all_volatility_data(time_frame)

    # Group by coin
    small_cap_df = vol_df[small_cap]
    mid_cap_df = vol_df[mid_cap]
    large_cap_df = vol_df[large_cap]

    # Create a figure
    fig = go.Figure()

    # Define function for adding percentile lines
    def add_percentiles(df, visible=False):
        _, q3, q1 = calculate_percentiles(df)
        x_range = [df.index.min(), df.index.max()]
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[q1, q1],
                mode="lines",
                name="25th percentile",
                line=dict(color="red", width=4),
                visible=visible,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[q3, q3],
                mode="lines",
                name="75th percentile",
                line=dict(color="green", width=4),
                visible=visible,
            )
        )

    # Add traces and percentile lines
    for coin, data in vol_df.items():
        fig.add_trace(
            go.Scatter(x=data.index, y=data, mode="lines", name=coin, visible=True)
        )
    add_percentiles(vol_df, visible=True)

    for coin, data in small_cap_df.items():
        fig.add_trace(
            go.Scatter(x=data.index, y=data, mode="lines", name=coin, visible=False)
        )
    if percentile_per_group:
        add_percentiles(small_cap_df)
    else:
        add_percentiles(vol_df)

    for coin, data in mid_cap_df.items():
        fig.add_trace(
            go.Scatter(x=data.index, y=data, mode="lines", name=coin, visible=False)
        )
    if percentile_per_group:
        add_percentiles(mid_cap_df)
    else:
        add_percentiles(vol_df)

    for coin, data in large_cap_df.items():
        fig.add_trace(
            go.Scatter(x=data.index, y=data, mode="lines", name=coin, visible=False)
        )
    if percentile_per_group:
        add_percentiles(large_cap_df)
    else:
        add_percentiles(vol_df)

    # Define buttons
    all_visibility = (
        [True] * (len(vol_df.columns) + 2)
        + [False] * (len(small_cap_df.columns) + 2)
        + [False] * (len(mid_cap_df.columns) + 2)
        + [False] * (len(large_cap_df.columns) + 2)
    )
    large_visibility = (
        [False] * (len(vol_df.columns) + 2)
        + [False] * (len(small_cap_df.columns) + 2)
        + [False] * (len(mid_cap_df.columns) + 2)
        + [True] * (len(large_cap_df.columns) + 2)
    )
    mid_visibility = (
        [False] * (len(vol_df.columns) + 2)
        + [False] * (len(small_cap_df.columns) + 2)
        + [True] * (len(mid_cap_df.columns) + 2)
        + [False] * (len(large_cap_df.columns) + 2)
    )
    small_visibility = (
        [False] * (len(vol_df.columns) + 2)
        + [True] * (len(small_cap_df.columns) + 2)
        + [False] * (len(mid_cap_df.columns) + 2)
        + [False] * (len(large_cap_df.columns) + 2)
    )

    buttons = [
        dict(label="All", method="update", args=[{"visible": all_visibility}]),
        dict(label="Large cap", method="update", args=[{"visible": large_visibility}]),
        dict(label="Mid cap", method="update", args=[{"visible": mid_visibility}]),
        dict(label="Small cap", method="update", args=[{"visible": small_visibility}]),
    ]

    # Add buttons to layout
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
        ]
    )

    fig.show()
