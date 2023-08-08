from darts import TimeSeries

# Local imports
from data.csv_data import read_csv
from config import test_percentage, n_periods


def get_train_test(
    coin="BTC",
    time_frame="1d",
    col="log returns",
):
    # Read data from a CSV file
    data = read_csv(coin, time_frame, [col]).dropna()
    data["date"] = data.index

    # Create a Darts TimeSeries from the DataFrame
    time_series = TimeSeries.from_dataframe(data, "date", col)

    # Set parameters for sliding window and periods
    test_size = int(len(time_series) / (1 / test_percentage - 1 + n_periods))
    train_size = int(test_size * (1 / test_percentage - 1))

    # Save the training and test sets as lists of TimeSeries
    train_set = []
    test_set = []
    full_set = []

    for i in range(n_periods):
        # The train start shifts by the test size each period
        train_start = i * test_size
        train_end = train_start + train_size

        train_set.append(time_series[train_start:train_end])
        test_set.append(time_series[train_end : train_end + test_size])

        # The whole timeseries of this period
        full_set.append(time_series[train_start : train_end + test_size])

    return train_set, test_set, full_set
