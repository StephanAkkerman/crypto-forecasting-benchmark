import os
import pandas as pd
from darts import TimeSeries

large_cap = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "MATIC"]
mid_cap = ["LINK", "ETC", "XLM", "LTC", "TRX", "ATOM", "XMR"]
small_cap = ["VET", "ALGO", "EOS", "CHZ", "IOTA", "NEO", "XTZ"]

all_coins = large_cap + mid_cap + small_cap

timeframes = ["1m", "15m", "4h", "1d"]

models = [
    "ARIMA",
    "RNN",
    "LSTM",
    "GRU",
    "TCN",
    "NBEATS",
    "TFT",
    "RANDOM FOREST",
    "XGBOOST",
    "LIGHTGBM",
    "NHITS",
    "TBATS",
    "PROPHET",
]


def read_csv(coin: str, timeframe: str, col_names: list = ["close"]):
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


def get_train_val(
    coin="BTC",
    time_frame="1d",
    n_periods=9,
    test_size_percentage=0.25,
    val_size_percentage=0.1,
    input_chunk_length=24,
):
    train_val_set = []

    train_set, _ = get_train_test(coin, time_frame, n_periods, test_size_percentage)

    # Calculate the validation size
    val_len = int(val_size_percentage * len(train_set[0]))
    end = len(train_set[0]) - val_len

    for period in range(n_periods):
        train_val_period = []
        for v in range(val_len):
            train = train_set[period][: end + v]
            # Add the input_chunk_length to the validation set
            # if -val_len + v + 1 != 0:
            #    val = train_set[period][
            #        -val_len + v - input_chunk_length : -val_len + v + 1
            #    ]
            # else:
            #    val = train_set[period][-val_len + v - input_chunk_length :]
            val = train_set[period][end + v - input_chunk_length : end + v + 1]

            train_val_period.append((train, val))

        train_val_set.append(train_val_period)

    return train_val_set
