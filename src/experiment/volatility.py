import config
from data_analysis.volatility import (
    get_all_volatility_data,
    get_percentile,
    get_volatility,
)
from experiment.utils import get_predictions


def volatility_map(coin, time_frame):
    vol = get_volatility(coin=coin, time_frame=time_frame)
    low, high = get_percentile(vol)

    # create a mask for each category
    mask_low = vol < low
    mask_high = vol > high
    mask_mid = (vol >= low) & (vol <= high)

    vol[mask_low] = "low"
    vol[mask_high] = "high"
    vol[mask_mid] = "mid"

    return vol


def all_volatility_maps():
    timeframes = ["1m", "15m", "4h", "1d"]
    volatility_data = {}

    for time_frame in timeframes:
        volatility_data[time_frame] = volatility_map(time_frame)


def get_test_train(coin, time_frame):
    # Get predictions
    _, trains, tests, rmses = get_predictions(
        model=config.log_returns_model,
        forecasting_model="ARIMA",
        coin=coin,
        time_frame=time_frame,
        concatenated=False,
    )

    vol = volatility_map(coin=coin, time_frame=time_frame)

    data = []

    for train, test, rmses in zip(trains, tests, rmses):
        # Get the volatility for the train and test data
        vol_train = vol.loc[train.start_time() : train.end_time()]
        vol_test = vol.loc[test.start_time() : test.end_time()]

        # Count the number of low, mid and high volatility periods
        train_counts = vol_train.value_counts()
        test_counts = vol_test.value_counts()

        # Add it to the data
        data.append((train_counts, test_counts, rmses))

    print(data)
