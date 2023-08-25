import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


def get_volatility_counts(vol_series):
    """Get the count of each volatility level ('low', 'mid', 'high') from a time series."""
    return vol_series.value_counts().to_dict()


def plot_rmse_vs_volatility(df):
    """Plot the RMSE against the train and test volatilities."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="main_train_volatility", y="rmse", hue="main_test_volatility", data=df
    )
    plt.title("Impact of Train and Test Volatility on RMSE")
    plt.show()


def coin_vol_data(
    model: str = config.log_returns_model, coin: str = "BTC", time_frame: str = "1d"
):
    """Analyze and plot the impact of volatility on RMSE for different train-test splits."""

    if model in [config.extended_model, config.extended_to_raw_model]:
        models = config.ml_models
    else:
        models = config.all_models

    vol_data = {}
    for forecasting_model in models:
        _, trains, tests, rmses = get_predictions(
            model=model,
            forecasting_model=forecasting_model,
            coin=coin,
            time_frame=time_frame,
            concatenated=False,
        )

        vol_map = volatility_map(coin=coin, time_frame=time_frame)
        forecasting_data = {}

        for i, (train, test, rmse) in enumerate(zip(trains, tests, rmses)):
            vol_train = vol_map.loc[train.start_time() : train.end_time()]
            vol_test = vol_map.loc[test.start_time() : test.end_time()]

            train_volatility_counts = get_volatility_counts(vol_train)
            test_volatility_counts = get_volatility_counts(vol_test)

            forecasting_data[i] = {
                "train": train_volatility_counts,
                "test": test_volatility_counts,
                "rmse": rmse,
            }
        vol_data[forecasting_model] = forecasting_data

    return vol_data


def tf_vol_data(time_frame):
    for model in [
        # config.log_returns_model,
        config.log_to_raw_model,
        config.extended_model,
        config.extended_to_raw_model,
        config.raw_model,
        config.raw_to_log_model,
        config.scaled_model,
        config.scaled_to_log_model,
        config.scaled_to_raw_model,
        config.scaled_to_raw_model,
    ]:
        save_loc = f"{config.volatility_dir}/{model}"

        vol_dict = {}
        for coin in config.all_coins:
            vol_dict[coin] = coin_vol_data(
                model=model, coin=coin, time_frame=time_frame
            )

        # Save the data as dataframe
        df = pd.DataFrame(vol_dict)

        # Create path
        os.makedirs(save_loc, exist_ok=True)

        # Save the dataframe as a CSV file
        df.to_csv(f"{save_loc}/vol_{time_frame}.csv")
        print(f"Saved {model} volatility data for {time_frame} time frame.")


def temp():
    return
    df_list = [
        {
            "index": idx,
            "main_train_volatility": max(data["train"], key=data["train"].get),
            "main_test_volatility": max(data["test"], key=data["test"].get),
            "rmse": data["rmse"],
        }
        for idx, data in analysis_data.items()
    ]

    df = pd.DataFrame(df_list)
    print(
        df.groupby(["main_train_volatility", "main_test_volatility"])["rmse"].describe()
    )

    plot_rmse_vs_volatility(df)
