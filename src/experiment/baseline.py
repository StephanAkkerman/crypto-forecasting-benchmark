import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import config
from experiment.rmse import read_rmse_csv, plot_rmse_heatmaps


def read_comparison_csv(pred: str, time_frame: str, avg: bool = True):
    df = pd.read_csv(
        f"{config.comparison_dir}/{pred}/comparison_{time_frame}.csv", index_col=0
    )

    # Convert string to list
    df = df.applymap(lambda x: x.strip("[]").split(", "))

    # Convert list of strings to list of floats
    df = df.applymap(lambda x: [float(i) for i in x])

    if avg:
        # Average the values in the lists
        df = df.applymap(lambda x: np.mean(x))

    return df


# Create data that compares models with ARIMA as percentage
def create_baseline_comparison(
    pred: str = config.log_returns_pred,
    time_frame: str = "1d",
    baseline_model: str = "ARIMA",
):
    """Compare the RMSE of the baseline model (ARIMA) to the other models and saves the data as .csv"""

    rmse_df = read_rmse_csv(pred, time_frame=time_frame)

    if pred == config.extended_pred:
        baseline_df = read_rmse_csv(config.log_returns_pred, time_frame=time_frame)[
            baseline_model
        ]
    elif pred == config.extended_to_raw_pred:
        # Could also try config.raw_pred
        baseline_df = read_rmse_csv(config.log_to_raw_pred, time_frame=time_frame)[
            baseline_model
        ]

    # Initialize an empty dictionary to hold percentual differences
    percentual_difference_dict = {}

    for column in rmse_df.columns:
        if column != baseline_model:
            percentual_difference_dict[column] = []

            for _, row in rmse_df.iterrows():
                if pred in [config.extended_pred, config.extended_to_raw_pred]:
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
        f"{config.comparison_dir}/{pred}/comparison_{time_frame}.csv"
    )


def create_all_baseline_comparison():
    for model in [
        config.log_returns_pred,
        config.log_to_raw_pred,
        config.extended_pred,
        config.extended_to_raw_pred,
        config.raw_pred,
        config.raw_to_log_pred,
        config.scaled_pred,
        config.scaled_to_log_pred,
        config.scaled_to_raw_pred,
        config.scaled_to_raw_pred,
    ]:
        # Create dir
        os.makedirs(f"{config.comparison_dir}/{model}", exist_ok=True)

        print(f"Creating baseline comparison data for {model}")
        for time_frame in config.timeframes:
            create_baseline_comparison(pred=model, time_frame=time_frame)


def get_all_baseline_comparison(pred: str = config.log_returns_pred, ignore_model=[]):
    dfs = []
    for time_frame in config.timeframes:
        comparison_df = read_comparison_csv(pred, time_frame=time_frame).drop(
            columns=ignore_model
        )
        dfs.append(comparison_df)
    return dfs


def baseline_comparison_heatmap(pred: str = config.log_returns_pred, ignore_model=[]):
    # visualize
    plot_rmse_heatmaps(
        get_all_baseline_comparison(pred, ignore_model=ignore_model),
        title=f"RMSE percentual comparison between forecasting models and baseline (ARIMA) model for {pred}",
        titles=[f"Time Frame: {tf}" for tf in config.timeframes],
        flip_colors=True,
        vmin=-3,
        vmax=3,
    )


def bar_plot(
    pred: str = config.log_returns_pred, ignore_model=[], ignore_outliers: bool = True
):
    """
    Plots the mean of the RMSE for each model in a grouped bar plot.

    Parameters
    ----------
    model : str, optional
        The model output to use, by default config.log_returns_pred
    """

    # Create a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes = axes.flatten()

    # Loop through the list of DataFrames and axes to create a grouped bar plot for each
    for i, (rmse_df, ax) in enumerate(
        zip(get_all_baseline_comparison(pred=pred, ignore_model=ignore_model), axes)
    ):
        sns.barplot(data=rmse_df, orient="h", palette="Set3", ax=ax)

        ax.set_xlabel("RMSE")
        ax.set_ylabel("Model")
        ax.set_title(f"Time Frame {config.timeframes[i]}")

        # Apply logarithmic scale if log_scale is True
        if ignore_outliers:
            # Calculate the 5th and 95th percentiles for the x-axis limits
            all_values = rmse_df.values.flatten()
            if i == 0 or i == 1:
                xmin = np.percentile(all_values, 25)
            else:
                xmin = np.percentile(all_values, 15)

            # Set the x-axis limits
            ax.set_xlim(xmin, all_values.max())

    plt.tight_layout()
    fig.subplots_adjust(top=0.925)
    fig.suptitle("Comparison of RMSE between ARIMA and other models")
    plt.show()


def box_plot(
    pred: str = config.log_returns_pred, ignore_model=[], ignore_outliers: bool = True
):
    """
    Plots the mean of the RMSE for each model in a boxplot.

    Parameters
    ----------
    model : str, optional
        The model output to use, by default config.log_returns_pred
    """
    # Create a grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes = axes.flatten()

    # Loop through the list of DataFrames and axes to create a grouped bar plot for each
    for i, (rmse_df, ax) in enumerate(
        zip(get_all_baseline_comparison(pred=pred, ignore_model=ignore_model), axes)
    ):
        sns.boxplot(data=rmse_df, orient="h", palette="Set3", ax=ax)

        ax.set_xlabel("RMSE")
        ax.set_ylabel("Model")
        ax.set_title(f"Time Frame {config.timeframes[i]}")

        # Apply logarithmic scale if log_scale is True
        if ignore_outliers:
            # Calculate the 5th and 95th percentiles for the x-axis limits
            all_values = rmse_df.values.flatten()
            if i == 0 or i == 1:
                xmin = np.percentile(all_values, 25)
            else:
                xmin = np.percentile(all_values, 15)

            # Set the x-axis limits
            ax.set_xlim(xmin, all_values.max())

    plt.tight_layout()
    fig.subplots_adjust(top=0.925)
    fig.suptitle("Comparison of RMSE between ARIMA and other models")
    plt.show()
