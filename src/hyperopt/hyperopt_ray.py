import os
import gc
import multiprocessing

import torch
from ray import tune

# https://docs.ray.io/en/latest/tune/api/suggestion.html
from ray.tune.search.skopt import SkOptSearch

# https://docs.ray.io/en/latest/tune/api/schedulers.html
# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html
from ray.tune.schedulers import ASHAScheduler
from darts.metrics import rmse, mae

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
    all_coins,
    timeframes,
)

from train_test import get_train_test


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

    # add default args to model_args
    if model_name not in ["Prophet", "TBATS"]:
        model_args.update(default_args)

    # The first 5 models are not classic ML models
    if model_name == "RandomForest":
        return RandomForest(**model_args)
    elif model_name == "XGB":
        return XGBModel(**model_args)
    elif model_name == "LightGBM":
        return LightGBMModel(**model_args)
    elif model_name == "Prophet":
        return Prophet(**model_args)
    elif model_name == "TBATS":
        return TBATS(**model_args)
    elif model_name == "NBEATS":
        return NBEATSModel(**model_args, model_name=model_name)
    elif model_name == "RNN":
        return RNNModel(**model_args, model_name=model_name)
    elif model_name == "LSTM":
        return RNNModel(**model_args, model_name=model_name, model="LSTM")
    elif model_name == "GRU":
        return RNNModel(**model_args, model_name=model_name, model="GRU")
    elif model_name == "TCN":
        return TCNModel(**model_args, model_name=model_name)
    elif model_name == "TFT":
        return TFTModel(**model_args, model_name=model_name)
    elif model_name == "NHiTS":
        return NHiTSModel(**model_args, model_name=model_name)
    else:
        raise ValueError(f"Model {model_name} is not supported.")


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
    model.fit(series=train_series[period][:-val_len])  # verbose=False

    retrain = False
    if model_name.startswith("Prophet") or model_name.startswith("TBATS"):
        retrain = True

    # Evaluate the model
    pred = model.historical_forecasts(
        series=train_series[period],
        start=len(train_series[period]) - val_len,
        forecast_horizon=1,  # 1 step ahead forecasting
        stride=1,  # 1 step ahead forecasting
        retrain=retrain,
        verbose=False,
    )

    # Calculate the metrics
    rmse_val = rmse(val, pred)
    tune.report(rmse=rmse_val, mae=mae(val, pred))

    # save predictions as file
    pred.pd_dataframe().to_csv(os.path.join(data_loc, f"{rmse_val}_pred.csv"))
    val.pd_dataframe().to_csv(os.path.join(data_loc, "val.csv"))

    # Delete objects to free up memory
    del model
    del pred


def hyperopt(
    train_series: list,
    model_name: str,
    period: int,
    coin: str,
    time_frame: str,
    num_samples: int,
    resources_per_trial: dict,
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

    # Do not add model unspecific parameters to regression models
    if model_name not in ["RandomForest", "XGB", "LightGBM", "Prophet", "TBATS"]:
        search_space.update(model_unspecific)

    # https://docs.ray.io/en/latest/tune/key-concepts.html#analysis
    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
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

    # Delete objects to free up memory
    del analysis
    gc.collect()

    # Could try: https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution

    # if gpu enabled in resources, show gpu memory usage


def get_resources(parallel_trials: int) -> dict:
    cores = multiprocessing.cpu_count()

    resources_per_trial = {
        "cpu": cores // parallel_trials,
    }

    # Get the name of the current device
    try:
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        resources_per_trial.update({"gpu": 1 / parallel_trials})
    except Exception:
        print("No GPU found, using CPU instead")

    # If the name is A100
    if "A100" in device_name:
        # Add the A100 accelerator
        resources_per_trial.update(
            {
                "custom_resources": {"accelerator_type:A100": 1 / parallel_trials},
            }
        )

    return resources_per_trial


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


def hyperopt_dataset(
    model_name: str,
    coin: str,
    time_frame: str,
    num_samples: int,
    resources_per_trial: dict,
):
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
    hyperopt(
        train_series, model_name, 0, coin, time_frame, num_samples, resources_per_trial
    )


def hyperopt_full(model_name: str, num_samples: int, parallel_trials: int):
    """
    Hyperparameter optimization for all datasets, for a given model.

    Parameters
    ----------
    model_name : str
        The name of the model to be used.
    num_samples : int
        The number of samples to be used.
    """
    # not_yet_done = "IOTA"
    # for coin in all_coins[all_coins.index(not_yet_done) :]:
    resources = get_resources(parallel_trials)

    print(
        f"Starting {num_samples} hyperparameter optimization trials, running {parallel_trials} trials in parallel with the following resources per trial:\n",
        resources,
    )

    # Loop over all coins and timeframes
    for coin in all_coins:
        for tf in timeframes:
            hyperopt_dataset(model_name, coin, tf, num_samples, resources)


if __name__ == "__main__":
    hyperopt_full(model_name="TBATS", num_samples=20, parallel_trials=3)
