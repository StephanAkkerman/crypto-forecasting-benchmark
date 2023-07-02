import os
from warnings import warn
import multiprocessing

import torch
from ray.tune import CLIReporter
from ray.util.accelerators import __all__ as accelerators

from config import (
    use_GPU,
    cpu_cores,
    timeframes,
    num_samples,
    results_folder,
)

from search_space import model_config, model_unspecific


def get_resources(model_name: str, parallel_trials: int) -> dict:
    if parallel_trials > num_samples:
        parallel_trials = num_samples
        warn("parallel_trials > num_samples, setting parallel_trials = num_samples")

    # These models use a lot of GPU resources
    if model_name == "TCN":
        parallel_trials = 1
    elif model_name == "NBEATS":
        parallel_trials = 3
    # These model do not need a lot of GPU resources
    elif model_name in ["RandomForest", "XGB", "LightGBM", "Prophet"]:
        parallel_trials = 20

    # Utilize all CPU cores if set to -1
    cores = multiprocessing.cpu_count()
    if cpu_cores != -1:
        if cpu_cores > cores:
            warn("You cannot use more cores than you have, setting cpu_cores = cores")
        else:
            cores = cpu_cores

    if cores < num_samples:
        warn("Cores < num_samples, no CPU cores will be used for the trials!")

    # CPU must be ints
    resources_per_trial = {
        "cpu": cores // parallel_trials,
    }

    # Get the name of the current device
    try:
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        if use_GPU:
            resources_per_trial.update({"gpu": 1 / parallel_trials})
    except Exception:
        print("No GPU found, using CPU instead")

    # Add accelerator type if it is available
    for accelerator in [a.split("_")[-1] for a in accelerators]:
        if accelerator in device_name:
            resources_per_trial.update(
                {
                    "custom_resources": {
                        f"accelerator_type:{accelerator}": 1 / parallel_trials
                    },
                }
            )

    print(
        f"Starting {num_samples} hyperparameter optimization trials, running {parallel_trials} trials in parallel with the following resources per trial:\n",
        resources_per_trial,
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
        if not os.path.exists(f"{results_folder}/{model_name}/{coin}/{tf}"):
            if not os.path.exists(f"{results_folder}/{model_name}/{coin}"):
                if not os.path.exists(f"{results_folder}/{model_name}"):
                    if not os.path.exists(results_folder):
                        os.makedirs(results_folder)
                    os.makedirs(f"{results_folder}/{model_name}")
                os.makedirs(f"{results_folder}/{model_name}/{coin}")
            os.makedirs(f"{results_folder}/{model_name}/{coin}/{tf}")


def get_reporter(model_name):
    return CLIReporter(
        parameter_columns=list(model_config[model_name].keys()),
        metric_columns=["loss", "mae", "rmse"],
    )


def get_search_space(model_name: str):
    # Add the unspecifc parameters
    search_space = model_config[model_name]

    # Do not add model unspecific parameters to regression models
    if model_name not in ["RandomForest", "XGB", "LightGBM", "Prophet", "TBATS", "TCN"]:
        search_space.update(model_unspecific)

    return search_space


def delete_config(save_loc: str):
    # Delete previous config.json files in save_loc
    if "config.json" in os.listdir(save_loc):
        os.remove(os.path.join(save_loc, "config.json"))
