from darts import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt

# Local imports
from data.csv_data import read_csv
from data.vars import all_coins


# Load your data (replace this with your actual data)
def get_train_test(coin="BTC", time_frame="1d", n_periods=9, test_size_percentage=0.25):
    # Read data from a CSV file
    data = read_csv(coin, time_frame, ["log returns"]).dropna()
    data["date"] = data.index

    # Create a Darts TimeSeries from the DataFrame
    time_series = TimeSeries.from_dataframe(data, "date", "log returns")

    # Set parameters for sliding window and periods
    test_size = int(len(time_series) / (1 / test_size_percentage - 1 + n_periods))
    train_size = int(test_size * (1 / test_size_percentage - 1))

    print("Train size:", train_size)
    print("Test size:", test_size)

    # Save the training and test sets as lists of TimeSeries
    train_set = []
    test_set = []

    for i in range(n_periods):
        # The train start shifts by the test size each period
        train_start = i * test_size
        train_end = train_start + train_size

        train_set.append(time_series[train_start:train_end])
        test_set.append(time_series[train_end : train_end + test_size])

    return train_set, test_set


def plot_periods(
    timeframe="1d", n_periods=9, test_size_percentage=0.25, col_name="volatility"
):
    """
    Plots the volatility of all cryptocurrencies and the average volatility.
    Also shows the train and test sets for each period.

    Parameters
    ----------
    timeframe : str, optional
        The time frame to use, by default "1d"
    n_periods : int, optional
        The number of periods to plot, by default 9
    test_size_percentage : float, optional
        The percentage of the data to use for testing, by default 0.25
    """

    complete_df = pd.DataFrame()

    for coin in all_coins:
        coin_df = read_csv(
            coin=coin, timeframe=timeframe, col_names=[col_name]
        ).dropna()

        # Set the index to the dates from coin_df
        if complete_df.empty:
            complete_df.index = coin_df.index

        complete_df[coin] = coin_df[col_name].tolist()

    ax = complete_df.plot(figsize=(12, 6), alpha=0.3, legend=False)

    # Calculate the average of all volatilities
    avg_volatility = complete_df.mean(axis=1)

    # Plot the average volatility as a big red line with increased width
    avg_line = plt.plot(
        avg_volatility, color="red", linewidth=2, label=f"Average {col_name}"
    )

    # Calculate the overall average of the avg_volatility and plot it as a horizontal blue line
    overall_avg_volatility = avg_volatility.mean()
    overall_avg_line = plt.axhline(
        y=overall_avg_volatility,
        color="blue",
        linewidth=2,
        label=f"Overall Average {col_name}",
    )

    ts_length = 999
    test_size = int(ts_length / (1 / test_size_percentage - 1 + n_periods))
    train_size = int(test_size * (1 / test_size_percentage - 1))

    print("Train size:", train_size)
    print("Test size:", test_size)

    line_start = 1.7
    for i in range(n_periods):
        # The train start shifts by the test size each period
        train_start = i * test_size
        train_end = train_start + train_size

        # Show the train and test start and end of each period as horizontal lines near the top
        # Use the complete_df index to get the corresponding dates
        date_min = complete_df.index.min()
        date_max = complete_df.index.max()

        start_pos = (complete_df.index[train_start] - date_min) / (date_max - date_min)
        end_pos = (complete_df.index[train_end] - date_min) / (date_max - date_min)
        test_pos = (complete_df.index[min(train_end + test_size, 969)] - date_min) / (
            date_max - date_min
        )

        plt.axhline(y=line_start, xmin=start_pos, xmax=end_pos, color="blue")
        plt.axhline(y=line_start, xmin=end_pos, xmax=test_pos, color="red")
        line_start -= 0.1

    # Show legends only for the average volatility and overall average volatility lines
    ax.legend(handles=[avg_line[0], overall_avg_line], loc="best")

    # Set y-axis title
    ax.set_ylabel("Volatility")
    ax.set_xlabel("Date")

    plt.savefig(f"data/plots/{test_size}test_{n_periods}periods.png")
    plt.show()
