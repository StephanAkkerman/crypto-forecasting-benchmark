import os
import pandas as pd
from darts import TimeSeries

from config import test_percentage, n_periods


def read_csv(coin: str, timeframe: str, col_names: list = ["log returns"]):
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up two levels to the parent directory
    crypto_forecasting_folder = os.path.dirname(os.path.dirname(current_dir))

    # Go to the data folder
    data_folder = os.path.join(crypto_forecasting_folder, "data")

    # Go to the coins folder
    coins_folder = os.path.join(data_folder, "coins")

    df = pd.read_csv(f"{coins_folder}/{coin}/{coin}USDT_{timeframe}.csv")

    # Set date as index
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)

    return df[col_names]


def get_train_test(coin="BTC", time_frame="1d"):
    # Read data from a CSV file
    data = read_csv(coin, time_frame, ["log returns"]).dropna()
    data["date"] = data.index

    # Create a Darts TimeSeries from the DataFrame
    time_series = TimeSeries.from_dataframe(data, "date", "log returns")

    # Set parameters for sliding window and periods
    test_size = int(len(time_series) / (1 / test_percentage - 1 + n_periods))
    train_size = int(test_size * (1 / test_percentage - 1))

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
