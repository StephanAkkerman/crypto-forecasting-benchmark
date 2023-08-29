import os

import numpy as np
import pandas as pd

import config
from experiment.rmse import read_rmse_csv, plot_rmse_heatmap


# Create data that compares models with ARIMA as percentage
def create_baseline_comparison(
    model: str = config.log_returns_model,
    time_frame: str = "1d",
    baseline_model: str = "ARIMA",
):
    """Compare the RMSE of the baseline model (ARIMA) to the other models."""

    rmse_df = read_rmse_csv(model, time_frame=time_frame)

    if model == config.extended_model:
        baseline_df = read_rmse_csv(config.log_returns_model, time_frame=time_frame)[
            baseline_model
        ]
    elif model == config.extended_to_raw_model:
        # Could also try config.raw_model
        baseline_df = read_rmse_csv(config.log_to_raw_model, time_frame=time_frame)[
            baseline_model
        ]

    # Initialize an empty dictionary to hold percentual differences
    percentual_difference_dict = {}

    for column in rmse_df.columns:
        if column != baseline_model:
            percentual_difference_dict[column] = []

            for _, row in rmse_df.iterrows():
                if model in [config.extended_model, config.extended_to_raw_model]:
                    baseline = np.array(baseline_df.loc[row.name])
                else:
                    baseline = np.array(row[baseline_model])

                model_values = np.array(row[column])

                percentual_difference = ((baseline - model_values) / baseline) * 100

                percentual_difference_dict[column].append(list(percentual_difference))

    percentual_difference_df = pd.DataFrame(
        percentual_difference_dict, index=rmse_df.index
    )

    # Save the data to csv
    percentual_difference_df.to_csv(
        f"{config.comparison_dir}/{model}/comparison_{time_frame}.csv"
    )


def create_all_baseline_comparison():
    for model in [
        # config.log_returns_model,
        # config.log_to_raw_model,
        config.extended_model,
        config.extended_to_raw_model,
        # config.raw_model,
        # config.raw_to_log_model,
        # config.scaled_model,
        # config.scaled_to_log_model,
        # config.scaled_to_raw_model,
        # config.scaled_to_raw_model,
    ]:
        # Create dir
        os.makedirs(f"{config.comparison_dir}/{model}", exist_ok=True)

        print(f"Creating baseline comparison data for {model}")
        for time_frame in config.timeframes:
            create_baseline_comparison(model=model, time_frame=time_frame)


def baseline_comparison(
    model: str = config.log_returns_model, baseline_model: str = "ARIMA"
):
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

    # Convert to pd dataframe
    percentual_difference = pd.DataFrame(percentual_difference)

    # Add columns and indices
    percentual_difference.index = rmse_df.index
    percentual_difference.columns = rmse_df.columns

    # visualize
    plot_rmse_heatmap(
        percentual_difference,
        title=f"RMSE percentual comparison between {model} model and ARIMA model for 1d time frame",
        flip_colors=True,
        vmin=-3,
        vmax=3,
    )
