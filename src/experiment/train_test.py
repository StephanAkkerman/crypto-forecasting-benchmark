from darts import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from darts.dataprocessing.transformers import Scaler

# Local imports
from data.csv_data import read_csv
from config import test_percentage, n_periods


def get_train_test(
    coin="BTC",
    time_frame="1d",
    col="log returns",
    scale: bool = False,
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

        train = time_series[train_start:train_end]
        test = time_series[train_end : train_end + test_size]
        full = time_series[train_start : train_end + test_size]

        # If scale is True, scale the data
        if scale:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            transformer = Scaler(scaler)

            # Fit only on the training data
            transformer.fit(train)

            # Transform both train and test data
            train = transformer.transform(train)
            test = transformer.transform(test)
            full = transformer.transform(full)

        train_set.append(train)
        test_set.append(test)

        # The whole timeseries of this period
        full_set.append(full)

    return train_set, test_set, full_set
