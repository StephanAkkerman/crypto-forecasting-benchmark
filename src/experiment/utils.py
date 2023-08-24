import os

import numpy as np
import pandas as pd
from darts.metrics import rmse
from darts import concatenate
from darts.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from darts.dataprocessing.transformers import Scaler

# Local imports
import config
from config import (
    all_coins,
    timeframes,
    all_models,
    ml_models,
    model_output_dir,
    log_returns_model,
    raw_model,
    extended_model,
    scaled_model,
    log_to_raw_model,
    scaled_to_raw_model,
    extended_to_raw_model,
    scaled_to_log_model,
    raw_to_log_model,
)
from experiment.train_test import get_train_test
from data.csv_data import read_csv


def all_model_predictions(
    model: str, coin: str, time_frame: str
) -> (dict, pd.DataFrame):
    # Save the predictions and tests for each model
    model_predictions = {}

    if model == extended_model:
        models = ml_models
    else:
        models = all_models

    for model_name in models:
        preds, _, tests, rmses = get_predictions(
            model=model,
            forecasting_model=model_name,
            coin=coin,
            time_frame=time_frame,
        )
        # If the model does not exist, skip it
        if preds is not None:
            model_predictions[model_name] = (preds, tests, rmses)

    # Only use the third value in the tuple (the rmse) and convert to a dict
    rmses = {model: rmse for model, (_, _, rmse) in model_predictions.items()}
    rmse_df = pd.DataFrame(rmses)

    return model_predictions, rmse_df


def get_predictions(
    model: str,
    forecasting_model: str,
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
    forecasting_model : str
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
    trains = []
    tests = []
    rmses = []

    value_cols = ["log returns"]

    if model in [
        config.raw_model,
        config.extended_to_raw_model,
        config.log_to_raw_model,
        config.scaled_to_raw_model,
    ]:
        value_cols = ["close"]

    for period in range(config.n_periods):
        file_loc = f"{model_output_dir}/{model}/{forecasting_model}/{coin}/{time_frame}"
        pred_path = f"{file_loc}/pred_{period}.csv"
        train_path = f"{file_loc}/train_{period}.csv"
        test_path = f"{file_loc}/test_{period}.csv"
        if not os.path.exists(pred_path):
            print(f"Warning the following file does not exist: {pred_path}")
            return None, None, None

        # Create the prediction TimeSeries
        pred = TimeSeries.from_dataframe(
            pd.read_csv(pred_path), time_col="time", value_cols=value_cols
        )
        train = TimeSeries.from_dataframe(
            pd.read_csv(train_path), time_col="date", value_cols=value_cols
        )
        test = TimeSeries.from_dataframe(
            pd.read_csv(test_path), time_col="date", value_cols=value_cols
        )

        # Calculate the RMSE for this period and add it to the list
        rmses.append(rmse(test, pred))

        # Add it to list
        preds.append(pred)
        trains.append(train)
        tests.append(test)

    # Make it one big TimeSeries
    if model != extended_model and concatenated:
        preds = concatenate(preds, axis=0)
        trains = concatenate(trains, axis=0)
        tests = concatenate(tests, axis=0)

    return preds, trains, tests, rmses


def unscale_model():
    # Create scaled_to_log model data
    for forecasting_model in all_models:
        for coin in all_coins:
            print("Unscaling log returns for", forecasting_model, coin)
            for time_frame in timeframes:
                scaled_to_log(
                    model=scaled_model,
                    forecasting_model=forecasting_model,
                    coin=coin,
                    time_frame=time_frame,
                )


def scaled_to_log(model: str, forecasting_model: str, coin: str, time_frame: str):
    preds, _, _, _ = get_predictions(
        model=model,
        forecasting_model=forecasting_model,
        coin=coin,
        time_frame=time_frame,
        concatenated=False,
    )

    # Get the log data
    trains, tests, _ = get_train_test(coin=coin, time_frame=time_frame, scale=False)

    # Create a directory to save the predictions
    save_loc = f"{model_output_dir}/{scaled_to_log_model}/{forecasting_model}/{coin}/{time_frame}"
    os.makedirs(save_loc, exist_ok=True)

    # Loop over both lists
    for i, (pred, train, test) in enumerate(zip(preds, trains, tests)):
        # Build the scaler
        scaler = MinMaxScaler(feature_range=(-1, 1))
        transformer = Scaler(scaler)

        # Fit only on the training data
        transformer.fit(train)

        # Unscale the data
        pred_unscaled = transformer.inverse_transform(pred)

        # Convert to a dataframe
        pred_unscaled = pred_unscaled.pd_dataframe()
        test = test.pd_dataframe()

        # Save it as a .csv
        pred_unscaled.to_csv(f"{save_loc}/pred_{i}.csv")
        test.to_csv(f"{save_loc}/test_{i}.csv")


def raw_model_to_log():
    # Create raw_to_log model data
    for forecasting_model in all_models:
        for coin in all_coins:
            print("Converting price data to log returns for", forecasting_model, coin)
            for time_frame in timeframes:
                raw_to_log(
                    model=raw_model,
                    forecasting_model=forecasting_model,
                    coin=coin,
                    time_frame=time_frame,
                )


def raw_to_log(model: str, forecasting_model: str, coin: str, time_frame: str):
    preds, _, tests, _ = get_predictions(
        model=model,
        forecasting_model=forecasting_model,
        coin=coin,
        time_frame=time_frame,
        concatenated=False,
    )

    # Create a directory to save the predictions
    save_loc = (
        f"{model_output_dir}/{raw_to_log_model}/{forecasting_model}/{coin}/{time_frame}"
    )
    os.makedirs(save_loc, exist_ok=True)

    # Loop over both lists
    for i, (pred, test) in enumerate(zip(preds, tests)):
        # Convert pred and test to df
        pred = pred.pd_dataframe()
        test = test.pd_dataframe()

        # Add the last row of the previous pred and test at the front
        if i > 0:
            prev_pred = preds[i - 1].pd_dataframe()
            prev_test = tests[i - 1].pd_dataframe()

            # Add last row of previous pred
            pred = pd.concat([prev_pred.iloc[-1:], pred])
            test = pd.concat([prev_test.iloc[-1:], test])

        # Convert to log returns
        pred["log returns"] = np.log(pred["close"]).diff()
        test["log returns"] = np.log(test["close"]).diff()

        # Drop the first row -> NaN
        pred = pred.iloc[1:]
        test = test.iloc[1:]

        # Drop close column
        pred = pred.drop(columns=["close"])
        test = test.drop(columns=["close"])

        # Save it as a .csv
        pred.to_csv(f"{save_loc}/pred_{i}.csv")
        test.to_csv(f"{save_loc}/test_{i}.csv")


def all_log_models_to_price():
    for model in [log_returns_model, scaled_to_log_model, extended_model]:
        log_model_to_price(model=model)


def log_model_to_price(model: str = log_returns_model):
    if model == extended_model:
        models = ml_models

    if model in [log_returns_model, scaled_to_log_model]:
        # Scaled to log returns can be converted to raw
        models = all_models

    for forecasting_model in models:
        for coin in all_coins:
            print("Converting log returns to price for", forecasting_model, coin)
            for time_frame in timeframes:
                log_returns_to_price(
                    model=model,
                    forecasting_model=forecasting_model,
                    coin=coin,
                    time_frame=time_frame,
                )


def log_returns_to_price(
    model: str, forecasting_model: str, coin: str, time_frame: str
):
    """
    Convert a series of logarithmic returns to price series.

    Parameters
    ----------
    model : str
        The model that was used to predict the log returns
    forecasting_model : str
        The name of the forecasting model to get the predictions from
    coin : str
        The coin to get the predictions for
    time_frame : str
        The time frame to get the predictions for
    """
    preds, _, _, _ = get_predictions(
        model=model,
        forecasting_model=forecasting_model,
        coin=coin,
        time_frame=time_frame,
        concatenated=False,
    )

    if model == log_returns_model:
        transformed_model = log_to_raw_model
    elif model == extended_model:
        transformed_model = extended_to_raw_model
    elif model == scaled_to_log_model:
        transformed_model = scaled_to_raw_model
    else:
        print("This model cannot be converted to price.")
        return

    # Get the price of test data
    price_df = read_csv(coin=coin, timeframe=time_frame, col_names=["close"])

    # Create a directory to save the predictions
    save_loc = f"{model_output_dir}/{transformed_model}/{forecasting_model}/{coin}/{time_frame}"
    os.makedirs(save_loc, exist_ok=True)

    for i, prediction in enumerate(preds):
        # Check if the prediction is empty
        if f"{save_loc}/pred_{i}.csv" in os.listdir(save_loc):
            continue

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
        close.to_csv(f"{save_loc}/pred_{i}.csv")
        test.to_csv(f"{save_loc}/test_{i}.csv")

        # Reset close list
        close = []
