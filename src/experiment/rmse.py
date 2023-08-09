import os

import pandas as pd
from darts.metrics import rmse
from darts.timeseries import TimeSeries

# Local imports
from config import all_coins, timeframes, all_models, rmse_dir
from experiment.utils import all_model_predictions


def read_rmse_csv(model_dir: str, time_frame: str) -> pd.DataFrame:
    df = pd.read_csv(f"data/analysis/{model_dir}/rmse_{time_frame}.csv", index_col=0)

    # Convert string to list of floats
    df = df.applymap(lambda x: x.strip("[]").split(", "))

    # Convert list of strings to list of floats
    df = df.applymap(lambda x: [float(i) for i in x])

    return df


def build_rmse_database(model_dir: str = "models", skip_existing: bool = True):
    os.makedirs(f"{rmse_dir}/{model_dir}", exist_ok=True)

    for tf in timeframes:
        # Skip if the file already exists
        if skip_existing:
            if os.path.exists(f"{rmse_dir}/{model_dir}/rmse_{tf}.csv"):
                print(
                    f"{rmse_dir}/{model_dir}/rmse_{tf}.csv already exists, skipping..."
                )
                continue

        print(f"Building {rmse_dir}/{model_dir}/rmse_{tf}.csv...")

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
        rmse_df.to_csv(f"{rmse_dir}/{model_dir}/rmse_{tf}.csv", index=True)

        # Print number on Nan values
        nan_values = rmse_df.isna().sum().sum()
        if nan_values > 0:
            print(f"Number of NaN values in {tf} for {model_dir}: {nan_values}")


def build_all_rmse_databases():
    # Cannot be done for extended_models
    for model_dir in ["models", "raw_models", "extended_models"]:
        build_rmse_database(model_dir=model_dir)


def build_rmse_transformed(skip_existing: bool = False):
    model_dir = "transformed_models"
    os.makedirs(f"{rmse_dir}/{model_dir}", exist_ok=True)

    # TODO
