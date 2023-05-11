from darts import TimeSeries

from data.csv_data import read_csv


# Load your data (replace this with your actual data)
def get_train_test(coin="BTC", time_frame="1d", n_periods=9, test_size_percentage=0.25):
    trains = []
    tests = []

    # Read data from a CSV file
    data = read_csv(coin, time_frame, ["log returns"]).dropna()
    data["date"] = data.index

    # Create a Darts TimeSeries from the DataFrame
    time_series = TimeSeries.from_dataframe(data, "date", "log returns")

    # Set parameters for sliding window and periods
    test_size = int(len(time_series) / (1 / test_size_percentage - 1 + n_periods))
    window_size = int(test_size * (1 / test_size_percentage - 1))

    for i in range(n_periods):
        start = i * test_size
        # train_data = time_series.slice(start, start + window_size)
        train_data = time_series[start : start + window_size]
        test_data = time_series[start + window_size : start + window_size + test_size]

        # print(start, start + window_size, start + window_size + test_size)
        trains.append(train_data)
        tests.append(test_data)

    return time_series, trains, tests
