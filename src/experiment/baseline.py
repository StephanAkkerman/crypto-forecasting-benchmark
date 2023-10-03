import os
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy.stats import ttest_rel

import config
from experiment.rmse import read_rmse_csv, plot_rmse_heatmaps, plot_rmse_heatmap


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


def get_all_baseline_comparison(
    pred: str = config.log_returns_pred, ignore_model=[], trans: bool = False
):
    dfs = []
    for time_frame in config.timeframes:
        comparison_df = read_comparison_csv(pred, time_frame=time_frame).drop(
            columns=ignore_model
        )
        if trans:
            comparison_df = comparison_df.T

        dfs.append(comparison_df)
    return dfs


def baseline_comparison_heatmap(pred: str = config.log_returns_pred, ignore_model=[]):
    titles = [
        "One-Minute Time Frame",
        "Fifteen-Minute Time Frame",
        "Four-Hour Time Frame",
        "One-Day Time Frame",
    ]

    # visualize
    plot_rmse_heatmaps(
        get_all_baseline_comparison(pred, ignore_model=ignore_model, trans=False),
        title=f"RMSE percentual comparison between forecasting models and baseline (ARIMA) model for {pred}",
        titles=titles,
        flip_colors=True,
        vmin=-3,
        vmax=3,
        avg_y=False,
    )


def single_baseline_heatmap(pred: str = config.log_returns_pred):
    dfs = get_all_baseline_comparison(pred, ignore_model=[], trans=False)

    new_dfs = []

    # Calculate mean and keep it
    for df in dfs:
        # Calculate the mean of each column
        means = df.mean()
        new_dfs.append(means)

    # Merge into 1 df
    df = pd.concat(new_dfs, axis=1)
    df.columns = config.tf_names2

    plot_rmse_heatmap(
        df.T,
        title="",
        flip_colors=True,
        vmin=-1,
        vmax=1,
        avg_y=False,
        avg_x=False,
        x_label="Forecasting Model",
        y_label="Time Frame",
    )

def results_table(pred: str = config.log_returns_pred):
    dfs = get_all_baseline_comparison(pred, ignore_model=[], trans=False)

    new_dfs = []

    # Calculate mean and keep it
    for df in dfs:
        # Calculate the mean of each column
        means = df.mean()
        new_dfs.append(means)

    # Merge into 1 df
    df = pd.concat(new_dfs, axis=1)
    df.columns = config.tf_names2
    
    print(df)

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


def tf_correlation(pred: str = config.log_returns_pred):
    df_1m, df_15m, df_4h, df_1d = get_all_baseline_comparison(pred=pred)
    # Set time frame as a column
    df_1d["Time Frame"] = "1d"
    df_4h["Time Frame"] = "4h"
    df_15m["Time Frame"] = "15m"
    df_1m["Time Frame"] = "1m"

    # Concatenate the dataframes
    df_all = pd.concat([df_1d, df_4h, df_15m, df_1m])

    # Pivot the table so each row is a model and each column is a time frame
    df_pivot = df_all.set_index("Time Frame").T

    # Get all unique combinations of time frames
    time_frame_combinations = list(combinations(config.timeframes[::-1], 2))

    # Initialize an empty DataFrame to store correlations
    column_names = [f"{tf1}_{tf2}_correlation" for tf1, tf2 in time_frame_combinations]
    correlations = pd.DataFrame(index=df_pivot.index, columns=column_names)

    # Calculate correlation for each model between different time frames
    for model in df_pivot.index:
        for tf1, tf2 in time_frame_combinations:
            col_name = f"{tf1}_{tf2}_correlation"
            correlations.loc[model, col_name] = pearsonr(
                df_pivot.loc[model, tf1].values, df_pivot.loc[model, tf2].values
            )[
                0
            ]  # [0] is to get only the correlation coefficient, without p-value

    print(correlations)


def tf_significance(pred: str = config.log_returns_pred):
    df_1m, df_15m, df_4h, df_1d = get_all_baseline_comparison(pred=pred)
    # Set time frame as a column
    df_1d["Time Frame"] = "1d"
    df_4h["Time Frame"] = "4h"
    df_15m["Time Frame"] = "15m"
    df_1m["Time Frame"] = "1m"

    # Concatenate the dataframes
    df_all = pd.concat([df_1d, df_4h, df_15m, df_1m])

    # Pivot the table so each row is a model and each column is a time frame
    df_pivot = df_all.set_index("Time Frame").T

    # Get all unique combinations of time frames
    time_frame_combinations = list(combinations(config.timeframes[::-1], 2))

    # Initialize an empty DataFrame to store correlations
    column_names = [f"{tf1}_{tf2}_significance" for tf1, tf2 in time_frame_combinations]
    correlations = pd.DataFrame(index=df_pivot.index, columns=column_names)

    # Calculate correlation for each model between different time frames
    for model in df_pivot.index:
        for tf1, tf2 in time_frame_combinations:
            col_name = f"{tf1}_{tf2}_significance"
            t_statistic, p_value = ttest_rel(
                df_pivot.loc[model, tf1].values, df_pivot.loc[model, tf2].values
            )
            # Significance level
            if p_value < 0.05:
                # First time frame > second time frame
                if np.mean(df_pivot.loc[model, tf1].values) > np.mean(
                    df_pivot.loc[model, tf2].values
                ):
                    result = f"{tf1}"
                else:
                    result = f"{tf2}"
            else:
                result = "None"

            correlations.loc[model, col_name] = result
    print(correlations)


def scaled_heatmap():
    dfs = []

    for time_frame in config.timeframes:
        # Compare log returns ARIMA to scaled forecasting models
        forecasting_df = read_rmse_csv(config.scaled_to_log_pred, time_frame=time_frame)
        baseline_df = read_rmse_csv(config.log_returns_pred, time_frame=time_frame)

        # Initialize an empty dictionary to hold percentual differences
        percentual_difference_dict = {}

        for column in forecasting_df.columns:
            if column != "ARIMA":
                percentual_difference_dict[column] = []

                for i, row in forecasting_df.iterrows():
                    baseline = np.array(baseline_df.loc[i]["ARIMA"])

                    model_values = np.array(row[column])

                    percentual_difference = ((baseline - model_values) / baseline) * 100

                    percentual_difference_dict[column].append(
                        round(np.mean(list(percentual_difference)))
                    )

        percentual_difference_df = pd.DataFrame(
            percentual_difference_dict, index=forecasting_df.index
        )
        dfs.append(percentual_difference_df)

    # Use values of scaled_to_log_pred for 1m and 15m time frame
    titles = [
        "One-Minute Time Frame",
        "Fifteen-Minute Time Frame",
        "Four-Hour Time Frame",
        "One-Day Time Frame",
    ]

    # visualize
    plot_rmse_heatmaps(
        dfs,
        title=f"RMSE percentual comparison between forecasting models and baseline (ARIMA) model for",
        titles=titles,
        flip_colors=True,
        vmin=-3,
        vmax=3,
        avg_y=False,
    )
