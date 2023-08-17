import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
import config
from config import (
    all_coins,
    timeframes,
    all_models,
    ml_models,
    rmse_dir,
    transformed_model,
    log_returns_model,
    extended_model,
    raw_model,
    rmse_dir,
    n_periods,
)
from experiment.utils import all_model_predictions


def read_rmse_csv(model: str, time_frame: str) -> pd.DataFrame:
    df = pd.read_csv(f"{rmse_dir}/{model}/rmse_{time_frame}.csv", index_col=0)

    # Convert string to list of floats
    df = df.applymap(lambda x: x.strip("[]").split(", "))

    # Convert list of strings to list of floats
    df = df.applymap(lambda x: [float(i) for i in x])

    return df


def build_rmse_database(model: str = log_returns_model, skip_existing: bool = True):
    os.makedirs(f"{rmse_dir}/{model}", exist_ok=True)

    for tf in timeframes:
        # Skip if the file already exists
        if skip_existing:
            if os.path.exists(f"{rmse_dir}/{model}/rmse_{tf}.csv"):
                print(f"{rmse_dir}/{model}/rmse_{tf}.csv already exists, skipping...")
                continue

        print(f"Building {rmse_dir}/{model}/rmse_{tf}.csv...")

        # Data will be added to this DataFrame
        rmse_df = pd.DataFrame()

        for coin in all_coins:
            # Get the predictions
            _, rmse_df_coin = all_model_predictions(
                model=model, coin=coin, time_frame=tf
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
        rmse_df.to_csv(f"{rmse_dir}/{model}/rmse_{tf}.csv", index=True)

        # Print number on Nan values
        nan_values = rmse_df.isna().sum().sum()
        if nan_values > 0:
            print(f"Number of NaN values in {tf} for {model}: {nan_values}")


def build_all_rmse_databases():
    # Cannot be done for extended_models
    for model_dir in [log_returns_model, raw_model, transformed_model]:
        build_rmse_database(model=model_dir)


def rmse_comparison(
    time_frame: str = "1d", model_1=transformed_model, model_2=raw_model
):
    # 1. Load the data
    rmse_1 = read_rmse_csv(model_1, time_frame)
    rmse_2 = read_rmse_csv(model_2, time_frame)

    # 2. Average the lists in the dataframe
    rmse_1 = rmse_1.applymap(lambda x: np.mean(x))
    rmse_2 = rmse_2.applymap(lambda x: np.mean(x))

    # 3. Calculate the percentual difference
    percentual_difference = ((rmse_2 - rmse_1) / rmse_1) * 100

    # Add average row at the bottom
    percentual_difference.loc["Average"] = percentual_difference.mean()

    # Add average column at the right
    percentual_difference["Average"] = percentual_difference.mean(axis=1)

    # 4. Display or save the resulting table
    print(percentual_difference)

    plot_rmse_heatmap(
        percentual_difference,
        title=f"RMSE percentual comparison between {model_1} model and {model_2} model for {time_frame} time frame",
        flip_colors=True,
    )

    # To save to a new CSV
    # percentual_difference.to_csv('percentual_difference.csv', index=False)


def extended_rmse_df(time_frame: str) -> pd.DataFrame:
    # Get RMSE data
    rmse_df = read_rmse_csv(model=config.extended_model, time_frame=time_frame)

    # Get the first value of each list in the dataframe -> period 0
    # Change the format that the first column is the period and forget about coin names
    data = {}
    for model in rmse_df.columns:
        # Get the RMSEs for the given model
        data[model] = rmse_df[model].iloc[: config.n_periods].tolist()

    return pd.DataFrame(data, index=range(config.n_periods))


def rmse_heatmap(time_frame: str, model=log_returns_model):
    if model == extended_model:
        rmse = extended_rmse_df(time_frame)
        decimals = 4
    else:
        rmse = read_rmse_csv(model, time_frame)
        decimals = 2

    # Round the values in the list
    rmse = rmse.applymap(lambda x: np.mean(x))

    # Add average column at the right
    rmse["Average"] = rmse.mean(axis=1)

    plot_rmse_heatmap(
        rmse,
        title=f"RMSE heatmap for {model} model for {time_frame} time frame",
        round_decimals=decimals,
    )


def plot_rmse_heatmap(
    df: pd.DataFrame,
    title: str,
    flip_colors: bool = False,
    round_decimals: int = 2,
    vmin: float = None,
    vmax: float = None,
):
    if flip_colors:
        cmap = "RdYlGn"
    else:
        # Green low, red high
        cmap = "RdYlGn_r"

    # Specify custom range
    if vmin == None:
        vmin = df.min().min()
    if vmax == None:
        vmax = df.max().max()

    plt.figure(figsize=(15, 10))
    plt.rcParams["axes.grid"] = False
    sns.heatmap(
        df,
        annot=True,
        cmap=cmap,
        fmt=f".{round_decimals}f",
        center=0,
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(title)
    plt.show()


def baseline_comparison(model: str = log_returns_model, baseline_model: str = "ARIMA"):
    """Compare the RMSE of the baseline model (ARIMA) to the other models."""

    rmse_df = read_rmse_csv(model, time_frame="1d")

    # Round the values in the list
    rmse_df = rmse_df.applymap(lambda x: np.mean(x))

    # get the baseline
    baseline = rmse_df[baseline_model]

    # drop the baseline
    rmse_df = rmse_df.drop(columns=[baseline_model])

    # Calculate the percentual difference
    # percentual_difference = ((rmse_df - baseline[:, None]) / baseline[:, None]) * 100
    # Higher is better
    percentual_difference = (
        (baseline.values[:, None] - rmse_df.values) / baseline.values[:, None]
    ) * 100

    # visualize
    plot_rmse_heatmap(
        percentual_difference,
        title=f"RMSE percentual comparison between {model} model and ARIMA model for 1d time frame",
        flip_colors=True,
        vmin=-3,
        vmax=3,
    )


def raw_to_log_returns():
    # Necessary to compare RMSE values in raw model, otherwise BTC will dominate
    pass
