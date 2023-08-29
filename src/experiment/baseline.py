import numpy as np

import config
from experiment.rmse import read_rmse_csv, plot_rmse_heatmap

# Create data that compares models with ARIMA as percentage


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

    # visualize
    plot_rmse_heatmap(
        percentual_difference,
        title=f"RMSE percentual comparison between {model} model and ARIMA model for 1d time frame",
        flip_colors=True,
        vmin=-3,
        vmax=3,
    )
