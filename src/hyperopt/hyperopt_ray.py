import os
import json

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
    all_coins,
    timeframes,
    models,
    num_samples,
    results_folder,
    parallel_trials,
    default_args,
    hyperopt_period,
)

from train_test import get_train_test
from utils import (
    create_dirs,
    get_resources,
    get_reporter,
    get_search_space,
    delete_config,
)


def get_model(model_name: str, model_args: dict):
    """
    Gets the model from the full_model_name and model_args.

    Parameters
    ----------
    model_name : str
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


def load_config(data_loc: str):
    # Load previously used configurations
    used_configs = []
    config_file = os.path.join(data_loc, "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            try:
                used_configs = json.load(f)
            except Exception as e:
                print("Error loading config file:", e)
                print("Config file:\n", config_file)

    return used_configs


def save_config(data_loc, used_configs, model_args):
    # Save the configuration after a successful trial
    config_file = os.path.join(data_loc, "config.json")
    try:
        used_configs.append(model_args)
        with open(config_file, "w") as f:
            json.dump(used_configs, f)
    except Exception as e:
        print(f"Could not save the configuration file: {e}")


def save_trial_results(data_loc, rmse_val, pred, val):
    # save predictions as file
    pred.pd_dataframe().to_csv(os.path.join(data_loc, f"{rmse_val}_pred.csv"))
    val.pd_dataframe().to_csv(os.path.join(data_loc, "val.csv"))


def train_model(
    model_args: dict,
    model_name: str,
    data_loc: str,
    period: int,
    train_series: list,
    save_results: bool = True,
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

    used_configs = load_config(data_loc)

    # If the current configuration has been used before, skip this trial
    if model_args in used_configs:
        tune.report(skip=True, rmse=999)  # This will mark the trial as completed
        return

    # Get the model object
    model = get_model(model_name.split("_")[0], model_args)

    # The validation set is 10% of the training set
    val_len = int(val_percentage * len(train_series[0]))
    val = train_series[period][-val_len:]

    # Train the model
    model.fit(series=train_series[period][:-val_len])  # verbose=False

    # Prophet and TBATS need to be retrained
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

    # Save the predictions and the validation set
    if save_results:
        save_trial_results(data_loc=data_loc, rmse_val=rmse_val, pred=pred, val=val)

    # Save the configuration
    save_config(data_loc=data_loc, used_configs=used_configs, model_args=model_args)

    # Report the results
    tune.report(rmse=rmse_val, mae=mae(val, pred))


def hyperopt(
    train_series: list,
    model_name: str,
    coin: str,
    time_frame: str,
    save_loc: str,
    save_results: bool = True,
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
    """
    train_fn_with_parameters = tune.with_parameters(
        train_model,
        model_name=f"{model_name}_{coin}_{time_frame}_{hyperopt_period}",
        data_loc=save_loc,
        period=hyperopt_period,
        train_series=train_series,
        save_results=save_results,
    )

    # https://docs.ray.io/en/latest/tune/key-concepts.html#analysis
    return tune.run(
        train_fn_with_parameters,
        resources_per_trial=get_resources(model_name, parallel_trials=parallel_trials),
        config=get_search_space(model_name),
        num_samples=num_samples,  # the number of combinations to try
        scheduler=ASHAScheduler(),
        metric="rmse",
        mode="min",  # "min" or "max
        progress_reporter=get_reporter(model_name),
        search_alg=SkOptSearch(),
        verbose=2,  # 0: silent, 1: only status updates, 2: status and trial results 3: most detailed
        local_dir="ray_results",
        name=f"{model_name}_{coin}_{time_frame}_{hyperopt_period}",  # folder in local_dir
        trial_name_creator=lambda trial: model_name,  # folder in name file
    )


def hyperopt_dataset(
    model_name: str,
    coin: str,
    time_frame: str,
    save_results: bool,
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
    """

    # load data
    train_series, _ = get_train_test(coin=coin, time_frame=time_frame)

    # Create the folders
    create_dirs(model_name, coin)

    # Create folder to save results
    folder_loc = f"{results_folder}/{model_name}/{coin}/{time_frame}"

    # Save the images in here
    save_loc = os.path.join(os.getcwd(), folder_loc)

    # Delete the config file if it exists
    delete_config(save_loc)

    analysis = hyperopt(
        train_series=train_series,
        model_name=model_name,
        coin=coin,
        time_frame=time_frame,
        save_loc=save_loc,
        save_results=save_results,
    )

    if save_results:
        analysis.results_df.to_csv(f"{folder_loc}/analysis.csv", index=False)


def hyperopt_model(model: str, save_results: bool):
    """
    Hyperparameter optimization for all datasets, for a given model.

    Parameters
    ----------
    model : str
        The name of the model to be used.
    save_results : bool
        Whether to save the results or not.
    """

    for coin in all_coins:
        for tf in timeframes:
            hyperopt_dataset(model, coin, tf, save_results)


def hyperopt_full(save_results: bool):
    """
    Hyperparameter optimization for all datasets, for all models.

    Parameters
    ----------
    save_results : bool
        Whether to save the results or not.
    """

    for model in models:
        # Prophet does not work on cluster :(
        if model == "Prophet":
            continue
        hyperopt_model(model, save_results)


if __name__ == "__main__":
    hyperopt_full(save_results=True)
    # hyperopt_model("TBATS", False)
