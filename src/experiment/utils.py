import os

import numpy as np
import pandas as pd
from darts.metrics import rmse
from darts import concatenate
from darts.timeseries import TimeSeries

# Local imports
from config import (
    all_coins,
    timeframes,
    all_models,
    ml_models,
    log_returns_model_dir,
    model_output_dir,
    transformed_model_dir,
    log_returns_model,
    raw_model,
    transformed_model,
    extended_model,
)
from data.csv_data import read_csv


def all_model_predictions(
    model_dir: str, coin: str, time_frame: str
) -> (dict, pd.DataFrame):
    # Save the predictions and tests for each model
    model_predictions = {}

    if model_dir == extended_model:
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


def get_predictions(
    model_dir: str,
    model_name: str,
    coin: str,
    time_frame: str,
    concatenated: bool = True,
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

    if model_dir in [log_returns_model, extended_model]:
        value_cols = ["log returns"]
    elif model_dir in [raw_model, transformed_model]:
        value_cols = ["close"]

    for period in range(5):
        file_path = f"{model_output_dir}/{model_dir}/{model_name}/{coin}/{time_frame}/pred_{period}.csv"
        test_path = f"{model_output_dir}/{model_dir}/{model_name}/{coin}/{time_frame}/test_{period}.csv"
        if not os.path.exists(file_path):
            print(
                f"Warning the following file does not exist: {model_output_dir}/{model_dir}/{model_name}/{coin}/{time_frame}/pred_{period}.csv"
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
    if model_dir != "extended_models" and concatenated:
        preds = concatenate(preds, axis=0)
        tests = concatenate(tests, axis=0)
    else:
        # All are the same
        tests = tests[0]

    return preds, tests, rmses


def all_log_returns_to_price(model_dir: str = log_returns_model):
    # These already use the close price
    if model_dir == raw_model:
        return

    if model_dir == extended_model:
        models = ml_models

    if model_dir == log_returns_model:
        models = all_models

    for model in models:
        for coin in all_coins:
            print("Converting log returns to price for", model, coin)
            for time_frame in timeframes:
                log_returns_to_price(
                    model_dir=model_dir, model=model, coin=coin, time_frame=time_frame
                )


def log_returns_to_price(model_dir: str, model: str, coin: str, time_frame: str):
    """Convert a series of logarithmic returns to price series."""
    preds, _, _ = get_predictions(
        model_dir=model_dir,
        model_name=model,
        coin=coin,
        time_frame=time_frame,
        concatenated=False,
    )

    # Get the price of test data
    price_df = read_csv(coin=coin, timeframe=time_frame, col_names=["close"])

    # Create a directory to save the predictions
    os.makedirs(f"{transformed_model_dir}/{model}/{coin}/{time_frame}", exist_ok=True)

    for i, prediction in enumerate(preds):
        # Start with 1 before the prediction
        start_pos = price_df.index.get_loc(prediction.start_time()) - 1
        end_pos = price_df.index.get_loc(prediction.end_time()) + 1
        sliced_price_df = price_df.iloc[start_pos:end_pos]

        # Get the first close price
        close = [sliced_price_df["close"].to_list()[0]]

        # Convert the log returns to price
        for value in prediction.values():
            close.append(close[-1] * np.exp(value[0]))

        # Convert to a dataframe
        close = pd.DataFrame(
            {"close": close},
            index=[sliced_price_df.index],
        )
        # Rename index to time to match with other data
        close = close.rename_axis("time")

        test = pd.DataFrame(
            {"close": sliced_price_df["close"].to_list()}, index=[sliced_price_df.index]
        )

        # Remove the first row
        close = close.iloc[1:]
        test = test.iloc[1:]

        # Save it as a .csv
        close.to_csv(
            f"{transformed_model_dir}/{model}/{coin}/{time_frame}/pred_{i}.csv"
        )
        test.to_csv(f"{transformed_model_dir}/{model}/{coin}/{time_frame}/test_{i}.csv")

        # Reset close list
        close = []
