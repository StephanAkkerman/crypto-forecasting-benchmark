import os
import glob
import matplotlib.pyplot as plt
import pandas as pd

from ray import tune
from darts.metrics import rmse, mae

# https://docs.ray.io/en/latest/tune/api/suggestion.html
from ray.tune.search.skopt import SkOptSearch

# https://docs.ray.io/en/latest/tune/api/schedulers.html
# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html
from ray.tune.schedulers import ASHAScheduler

# Models
from darts.models import (
    RNNModel,
    TCNModel,
    NBEATSModel,
    TFTModel,
    RandomForest,
    XGBModel,
    LightGBMModel,
    NHiTSModel,
    TBATS,
    Prophet,
)

# Local files
from config import (
    val_percentage,
    model_config,
    get_reporter,
    model_unspecific,
    default_args,
)

from train_test import get_train_test, all_coins, timeframes, models


def save_plot(save_loc: str):
    """
    Create a plot of the forecast and save it to save_loc.

    Parameters
    ----------
    save_loc : str
        The location to save the plot to.
    """

    # Get all .csv files in save_loc
    csv_files = glob.glob(os.path.join(save_loc, "*.csv"))

    predictions = []

    # Read all .csv files in file_loc
    for file in csv_files:
        file_name = file.split("\\")[-1]
        rmse = file_name.split("_")[0]

        data = pd.read_csv(file)

        # Set the validation data
        if file_name == "val.csv":
            val_data = data

        # If the file is a prediction file
        elif file_name.endswith("pred.csv"):
            predictions.append((data, rmse))

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    # Loop over items in dict
    for df, rmse in predictions:
        df["log returns"].plot(ax=ax, label=f"RMSE: {rmse}")
    # plt.plot(val_data, label="Test Set")
    val_data["log returns"].plot(ax=ax, label="Test Set")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Test Set vs. Forecast")
    plt.savefig(f"{save_loc}/forecast.png")
    plt.close()


def get_model(full_model_name: str, model_args: dict):
    """
    Gets the model from the full_model_name and model_args.

    Parameters
    ----------
    full_model_name : str
        This is the name of the model including the coin and timeframe.
    model_args : dict
        The arguments to pass to the model.

    Returns
    -------
    darts model
        Depending on the full_model_name, a different model will be returned.

    Raises
    ------
    ValueError
        If the model_name is not supported.
    """

    model_name = full_model_name.split("_")[0]

    if model_name not in models:
        raise ValueError(f"Model {model_name} not found in {models}")

    # add default args to model_args
    model_args.update(default_args)
    model_args.update({"model_name": model_name})

    if model_name == "NBEATS":
        return NBEATSModel(**model_args)


def train_model(
    model_args: dict,
    model_name: str,
    data_loc: str,
    period: int,
    train_series: list,
):
    """
    Train the model and report the results.

    Parameters
    ----------
    model_args : dict
        The arguments to pass to the model.
    model_name : str
        The name of the model.
    data_loc : str
        The location to save the results to.
    period : int
        The period to train the model on.
    train_series : list
        The training data.
    """

    # Get the model object
    model = get_model(model_name, model_args)

    # The validation set is 10% of the training set
    val_len = int(val_percentage * len(train_series[0]))
    val = train_series[period][-val_len:]

    # Train the model
    model.fit(series=train_series[period][:-val_len], verbose=False)

    # Evaluate the model
    pred = model.historical_forecasts(
        series=train_series[period],
        start=len(train_series[period]) - val_len,
        forecast_horizon=1,  # 1 step ahead forecasting
        stride=1,  # 1 step ahead forecasting
        retrain=False,
        verbose=False,
    )

    # Calculate the metrics
    rmse_val = rmse(val, pred)
    tune.report(rmse=rmse_val, mae=mae(val, pred))

    # save predictions as file
    pred.pd_dataframe().to_csv(os.path.join(data_loc, f"{round(rmse_val, 4)}_pred.csv"))
    val.pd_dataframe().to_csv(os.path.join(data_loc, "val.csv"))


def hyperopt(
    train_series: list,
    model_name: str,
    period: int,
    coin: str,
    time_frame: str,
    num_samples: int,
):
    """
    This function will optimize the hyperparameters of the model.

    Parameters
    ----------
    train_series : list
        All the training data.
    model_name : str
        The name of the model, e.g. NBEATS.
    period : int
        The period to train the model on.
    coin : str
        The name of the coin.
    time_frame : str
        The time frame of the data.
    num_samples : int
        The number of samples to use for the hyperparameter optimization.
    """

    # Create folder to save results
    folder_loc = f"hyperopt_results/{model_name}/{coin}/{time_frame}"

    # Save the images in here
    save_loc = os.path.join(os.getcwd(), folder_loc)

    train_fn_with_parameters = tune.with_parameters(
        train_model,
        model_name=f"{model_name}_{coin}_{time_frame}_{period}",
        data_loc=save_loc,
        period=period,
        train_series=train_series,
    )

    # Add the unspecifc parameters
    search_space = model_config[model_name]
    search_space.update(model_unspecific)

    # https://docs.ray.io/en/latest/tune/key-concepts.html#analysis
    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial={"cpu": 12, "gpu": 1},  # CPU number is the number of cores
        config=search_space,
        num_samples=num_samples,  # the number of combinations to try
        scheduler=ASHAScheduler(),
        metric="rmse",
        mode="min",  # "min" or "max
        progress_reporter=get_reporter(model_name),
        search_alg=SkOptSearch(),
        verbose=2,  # 0: silent, 1: only status updates, 2: status and trial results 3: most detailed
        local_dir="ray_results",
        name=f"{model_name}_{coin}_{time_frame}_{period}",  # folder in local_dir
        trial_name_creator=lambda trial: model_name,  # folder in name file
    )

    # Save the results
    analysis.results_df.to_csv(f"{folder_loc}/period{period}_results.csv", index=False)
    save_plot(save_loc)


def create_dirs(model_name: str, coin: str):
    """
    Create the directories to save the results in.

    Parameters
    ----------
    model_name : str
        The name of the model.
    coin : str
        The name of the coin.
    """
    for tf in timeframes:
        if not os.path.exists(f"hyperopt_results/{model_name}/{coin}/{tf}"):
            if not os.path.exists(f"hyperopt_results/{model_name}/{coin}"):
                if not os.path.exists(f"hyperopt_results/{model_name}"):
                    if not os.path.exists("hyperopt_results"):
                        os.makedirs("hyperopt_results")
                    os.makedirs(f"hyperopt_results/{model_name}")
                os.makedirs(f"hyperopt_results/{model_name}/{coin}")
            os.makedirs(f"hyperopt_results/{model_name}/{coin}/{tf}")


def hyperopt_dataset(model_name: str, coin: str, time_frame: str, num_samples: int):
    """
    Performs hyperparameter optimization for a given dataset.

    Parameters
    ----------
    model_name : str
        The name of the model to be used.
    coin : str
        The name of the coin to be used.
    time_frame : str
        The time frame to be used.
    num_samples : int
        The number of samples to be used.
    """

    # load data
    train_series, _ = get_train_test(coin=coin, time_frame=time_frame)

    # Create the folders
    create_dirs(model_name, coin)

    # Perform hyperparameter optimization for period 0
    hyperopt(train_series, model_name, 0, coin, time_frame, num_samples)


def hyperopt_full(model_name: str, num_samples: int):
    """
    Hyperparameter optimization for all datasets, for a given model.

    Parameters
    ----------
    model_name : str
        The name of the model to be used.
    num_samples : int
        The number of samples to be used.
    """
    for coin in all_coins:
        for tf in timeframes:
            hyperopt_dataset(model_name, coin, tf, num_samples)


if __name__ == "__main__":
    hyperopt_full("NBEATS", 20)
