import os

import pandas as pd

import config
from data_analysis.volatility import (
    get_tf_percentile,
    get_volatility,
)
from experiment.train_test import get_train_test


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
                train_volatilty.append(
                    get_volatility_class(
                        volatility=mean_vol_train,
                        percentile25=percentile25,
                        percentile75=percentile75,
                    )
                )

                test_volatilty.append(
                    get_volatility_class(
                        volatility=mean_vol_test,
                        percentile25=percentile25,
                        percentile75=percentile75,
                    )
                )

            # Create a dataframe for the coin
            coin_df = pd.DataFrame(
                data=[
                    {
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
