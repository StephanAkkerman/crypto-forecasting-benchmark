import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import config
from data_analysis.volatility import (
    get_tf_percentile,
    get_volatility,
)
from experiment.train_test import get_train_test
from experiment.rmse import read_rmse_csv


def get_volatility_class(volatility, percentile25, percentile75):
    if volatility < percentile25:
        return "low"
    elif volatility > percentile75:
        return "high"
    else:
        return "normal"


def model_volatility_data(model: str = config.log_returns_model):
    """Analyze and plot the impact of volatility on RMSE for different train-test splits."""
    save_loc = f"{config.volatility_dir}/{model}"
    os.makedirs(save_loc, exist_ok=True)

    for time_frame in config.timeframes:
        # Calculate the percentiles for this time frame
        percentile25, percentile75 = get_tf_percentile(time_frame=time_frame)

        volatility_df = pd.DataFrame()

        for coin in config.all_coins:
            # Get the volatility for the coin
            volatility = get_volatility(coin=coin, time_frame=time_frame)

            # Get the train and test times
            trains, tests, _ = get_train_test(coin=coin, time_frame=time_frame)

            # Save the data here
            train_volatilty_class = []
            test_volatilty_class = []
            train_volatilty = []
            test_volatilty = []

            # Loop over each period
            for train, test in zip(trains, tests):
                # Determine the train and test volatility
                vol_train = volatility.loc[train.start_time() : train.end_time()]
                vol_test = volatility.loc[test.start_time() : test.end_time()]

                # Calculate the mean volatility for train and test
                mean_vol_train = vol_train.mean().values[0]
                mean_vol_test = vol_test.mean().values[0]

                # Calculate the volatility class for train and test
                train_volatilty_class.append(
                    get_volatility_class(
                        volatility=mean_vol_train,
                        percentile25=percentile25,
                        percentile75=percentile75,
                    )
                )

                test_volatilty_class.append(
                    get_volatility_class(
                        volatility=mean_vol_test,
                        percentile25=percentile25,
                        percentile75=percentile75,
                    )
                )

                # Also add 2 columns without classification and just the mean number
                train_volatilty.append(mean_vol_train)
                test_volatilty.append(mean_vol_test)

            # Create a dataframe for the coin
            coin_df = pd.DataFrame(
                data=[
                    {
                        "train_volatility_class": train_volatilty_class,
                        "test_volatility_class": test_volatilty_class,
                        "train_volatility": train_volatilty,
                        "test_volatility": test_volatilty,
                    }
                ],
                index=[coin],
            )

            # Add to df
            volatility_df = pd.concat([volatility_df, coin_df])

        # Save: volatility classification
        volatility_df.to_csv(f"{save_loc}/vol_{time_frame}.csv")


def create_all_volatility_data():
    for model in [
        config.log_returns_model,
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
        model_volatility_data(model=model)


def read_volatility_csv(model: str, time_frame: str):
    df = pd.read_csv(
        f"{config.volatility_dir}/{model}/vol_{time_frame}.csv", index_col=0
    )

    # Convert string to list
    df = df.applymap(lambda x: x.strip("[]").split(", "))

    # Remove ' from string
    df = df.applymap(lambda x: [i.strip("'") for i in x])

    return df


def volatility_boxplot(model: str = config.log_returns_model, time_frame: str = "1d"):
    # Read the data
    rmse_df = read_rmse_csv(model, time_frame=time_frame)
    vol_df = read_volatility_csv(model, time_frame=time_frame)

    # Determine how the volatility affects the RMSE
    rmse = rmse_df["ARIMA"]

    # Add rmse to vol_df
    vol_df["rmse"] = rmse

    # Flatten the dataframe
    df_flat = vol_df.apply(lambda x: x.explode()).reset_index()

    grouped = df_flat.groupby(["train_volatility", "test_volatility"])

    # Print out the statistics for each group
    for name, group in grouped:
        print(f"Group: {name}")
        print("Statistics:", group["rmse"].describe())
        print("---")

    plt.figure(figsize=(12, 8))
    sns.boxplot(x="train_volatility", y="rmse", hue="test_volatility", data=df_flat)
    plt.title("Impact of Train and Test Volatility on RMSE")
    plt.show()


def volatility_heatmap(model: str = config.log_returns_model, time_frame: str = "1d"):
    # Read the data
    rmse_df = read_rmse_csv(model, time_frame=time_frame)
    vol_df = read_volatility_csv(model, time_frame=time_frame)

    # Determine how the volatility affects the RMSE
    rmse = rmse_df["ARIMA"]

    # Add rmse to vol_df
    vol_df["rmse"] = rmse

    # Flatten the dataframe
    df_flat = vol_df.apply(lambda x: x.explode()).reset_index()

    # Group by 'train_volatility' and 'test_volatility' and calculate the mean RMSE for each group
    grouped = df_flat.groupby(["train_volatility", "test_volatility"]).mean()

    # Create a pivot table for the heatmap
    pivot_table = grouped.pivot("train_volatility", "test_volatility", "RMSE")

    # Draw the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_table, annot=True, cmap="coolwarm", cbar_kws={"label": "Mean RMSE"}
    )
    plt.title("Impact of Train and Test Volatility on RMSE")
    plt.show()
