import os

import numpy as np
import pandas as pd
from darts.metrics import rmse
from darts import concatenate
from darts.timeseries import TimeSeries

# Local imports
from config import all_coins, timeframes, all_models, ml_models
from data.csv_data import read_csv


def read_rmse_csv(model_dir: str, time_frame: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/analysis/{model_dir}/rmse_{time_frame}.csv", index_col=0)

    # Convert string to list of floats
    df = df.applymap(lambda x: x.strip("[]").split(", "))

    # Convert list of strings to list of floats
    df = df.applymap(lambda x: [float(i) for i in x])

    return df


def all_model_predictions(
    model_dir: str, coin: str, time_frame: str
) -> (dict, pd.DataFrame):
    # Save the predictions and tests for each model
    model_predictions = {}

    if model_dir == "extended_models":
        models = ml_models
    else:
        models = all_models

    for model in models:
        preds, tests, rmses = get_predictions(
            model_dir=model_dir, model_name=model, coin=coin, time_frame=time_frame
        )
        # If the model does not exist, skip it
        if preds is not None:
            model_predictions[model] = (preds, tests, rmses)

    # Only use the third value in the tuple (the rmse) and convert to a dict
    rmses = {model: rmse for model, (_, _, rmse) in model_predictions.items()}
    rmse_df = pd.DataFrame(rmses)

    return model_predictions, rmse_df


def build_rmse_database(model_dir: str = "models", skip_existing: bool = True):
    os.makedirs(f"data/analysis/{model_dir}", exist_ok=True)

    for tf in timeframes:
        # Skip if the file already exists
        if skip_existing:
            if os.path.exists(f"data/analysis/{model_dir}/rmse_{tf}.csv"):
                print(
                    f"data/analysis/{model_dir}/rmse_{tf}.csv already exists, skipping..."
                )
                continue

        print(f"Building data/analysis/{model_dir}/rmse_{tf}.csv...")

        # Data will be added to this DataFrame
        rmse_df = pd.DataFrame()

        for coin in all_coins:
            # Get the predictions
            _, rmse_df_coin = all_model_predictions(
                model_dir=model_dir, coin=coin, time_frame=tf
            )
            # Convert the dataframe to a list of lists
            rmse_df_list = pd.DataFrame(
                {col: [rmse_df_coin[col].tolist()] for col in rmse_df_coin}
            )
            # Add the coin to the index
            rmse_df_list.index = [coin]
            # Add the data to the dataframe
            rmse_df = pd.concat([rmse_df, rmse_df_list])

        # Save the dataframe to a csv
        rmse_df.to_csv(f"data/analysis/{model_dir}/rmse_{tf}.csv", index=True)

        # Print number on Nan values
        nan_values = rmse_df.isna().sum().sum()
        if nan_values > 0:
            print(f"Number of NaN values in {tf} for {model_dir}: {nan_values}")


def build_all_rmse_databases():
    # Cannot be done for extended_models
    for model_dir in ["models", "raw_models", "extended_models"]:
        build_rmse_database(model_dir=model_dir)


def get_predictions(
    model_dir: str,
    model_name: str,
    coin: str,
    time_frame: str,
) -> (TimeSeries, TimeSeries, list):
    """
    Gets the predictions for a given model.

    Parameters
    ----------
    model_dir : str
        Options are: "models" or "raw_models"
    model_name : str
        Options are the models that were trained, for instance "ARIMA"
    coin : str
        This can be any of the 21 coins that were trained on
    time_frame : str
        Options are: "1m", "15m", "4h", and "1d"

    Returns
    -------
    preds, tests, rmses
        The forecast (prediction), actual values, and the rmse for each period
    """
    preds = []
    tests = []
    rmses = []

    if model_dir in ["models", "extended_models"]:
        value_cols = ["log returns"]
    elif model_dir == "raw_models":
        value_cols = ["close"]

    for period in range(5):
        file_path = (
            f"data/{model_dir}/{model_name}/{coin}/{time_frame}/pred_{period}.csv"
        )
        test_path = (
            f"data/{model_dir}/{model_name}/{coin}/{time_frame}/test_{period}.csv"
        )
        if not os.path.exists(file_path):
            print(
                f"Warning the following file does not exist: data/{model_dir}/{model_name}/{coin}/{time_frame}/pred_{period}.csv"
            )
            return None, None, None

        # Create the prediction TimeSeries
        pred = pd.read_csv(file_path)
        pred = TimeSeries.from_dataframe(pred, time_col="time", value_cols=value_cols)

        test = pd.read_csv(test_path)
        test = TimeSeries.from_dataframe(test, time_col="date", value_cols=value_cols)

        # Calculate the RMSE for this period and add it to the list
        rmses.append(rmse(test, pred))

        # Add it to list
        preds.append(pred)
        tests.append(test)

    # Make it one big TimeSeries
    if model_dir != "extended_models":
        preds = concatenate(preds, axis=0)
        tests = concatenate(tests, axis=0)
    else:
        # All are the same
        tests = tests[0]

    return preds, tests, rmses


def log_returns_to_price(model_dir, model, coin, time_frame):
    """Convert a series of logarithmic returns to price series."""
    preds, tests, rmses = get_predictions(
        model_dir=model_dir, model_name=model, coin=coin, time_frame=time_frame
    )
    price_df = read_csv(coin=coin, timeframe=time_frame, col_names=["close"])

    # Start with 1 before the prediction
    start_pos = price_df.index.get_loc(preds.start_time()) - 1
    end_pos = price_df.index.get_loc(preds.end_time()) + 1
    price_df = price_df.iloc[start_pos:end_pos]

    # Get the first close price
    prices = [price_df["close"].to_list()[0]]

    for value in preds.values():
        prices.append(prices[-1] * np.exp(value[0]))

    # Convert to a dataframe
    prices = pd.DataFrame({"price": prices}, index=[price_df.index])

    # Save it as a .csv
    # prices.to_csv(f"data/{model_dir}/{model}/{coin}/{time_frame}/prices.csv")
    print(prices)
