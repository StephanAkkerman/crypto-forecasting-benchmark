import os
import math
import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from hyperopt.config import timeframes, all_coins


def get_predictions(pred_loc):
    """Get the predictions and validation data made during determining the best hyperparameters for the model

    Parameters
    ----------
    pred_loc: str
        The location of the prediction files
    Returns
    -------
    list
        Validation data
    list
        Predictions
    """

    # Get all .csv files in pred_loc
    csv_files = glob.glob(os.path.join(pred_loc, "*.csv"))

    predictions = []

    # Read all .csv files in file_loc
    for file in csv_files:
        # Set the validation data
        if file.endswith("val.csv"):
            val_data = pd.read_csv(file)

        # If the file is a prediction file
        elif file.endswith("pred.csv"):
            # The last part of the file name
            rmse = file.split(time_frame)[-1]

            # Remove the first character + "_pred.csv"
            rmse = rmse[1:].split("_pred.csv")[0]

            # Round to 4 decimal places
            rmse = round(float(rmse), 4)

            # Skip if rmse is too high
            if rmse < 1:
                predictions.append((pd.read_csv(file), rmse))

    return val_data, predictions


def pred_plot(model_name: str, coin: str, time_frame: str, save: bool = True):
    """
    Create a plot of the forecast and save it to save_loc.

    Parameters
    ----------
    model_name : str
        The name of the model
    coin : str
        The name of the coin, i.e. BTC
    time_frame: str
        The time frame, options are [1m, 15m, 4h, 1d]
    """
    pred_loc = f"hyperopt_results/{model_name}/{coin}/{time_frame}/"
    val_data, predictions = get_predictions(pred_loc)

    # Plot the results
    _, ax = plt.subplots(figsize=(12, 6))

    # Plot the test set
    val_data["log returns"].plot(
        ax=ax,
        label="Validation Set",
        color="black",
        zorder=5,
        linewidth=2,
    )

    # Loop over items in dict
    for df, rmse in predictions:
        df["log returns"].plot(ax=ax, label=f"RMSE: {rmse}")
    # plt.plot(val_data, label="Test Set")

    # Create the legend
    ax.legend(
        loc="lower center",
        ncol=5,
    )

    plt.xlabel("Time Step")
    plt.ylabel("Logarithmic Returns")
    plt.title("Validation Set vs. Forecast")
    if save:
        plt.savefig(f"{pred_loc}/forecast.png")
    else:
        plt.show()
    plt.close()


def get_analysis(model_name, coin, time_frame, keep_mae=False):
    # Read the results
    save_loc = f"hyperopt_results/{model_name}/{coin}/{time_frame}/"

    # Get the analysis.csv
    results = pd.read_csv(f"{save_loc}analysis.csv")

    dropped_cols = [
        "time_this_iter_s",
        "trial_id",
        "logdir",
        "done",
        "training_iteration",
        "date",
        "timestamp",
        "time_total_s",
        "pid",
        "hostname",
        "node_ip",
        "time_since_restore",
        "iterations_since_restore",
        "experiment_tag",
        "config/output_chunk_length",
        "config/pl_trainer_kwargs/enable_progress_bar",
        "config/pl_trainer_kwargs/accelerator",
    ]

    if not keep_mae:
        dropped_cols.append("mae")

    # Only keep useful columns
    dropped_cols = [
        col for col in dropped_cols if col in results.columns
    ]  # Only keep columns that exist in the DataFrame
    results = results.drop(
        dropped_cols,
        axis=1,
    )

    # Rename columns starting with config/
    results.rename(columns=lambda x: x.replace("config/", ""), inplace=True)

    # Sort by rmse
    return results.sort_values(by=["rmse"])


def float_to_int(val: float):
    """
    Helper function to convert whole floats to ints

    Parameters
    ----------
    val : float
        The value to convert

    Returns
    -------
    float or int
        The converted value
    """
    if isinstance(val, str):
        return val
    elif math.isnan(val):
        return None
    elif isinstance(val, float) and val.is_integer():
        return int(val)
    return val


def best_hyperparameters(model_name, coin, time_frame):
    analysis = get_analysis(model_name, coin, time_frame)

    # Get the best parameters
    best = analysis.iloc[0]

    # Remove metrics
    best_config = best.drop(["rmse"])

    # Convert to dict with correct types
    return {key: float_to_int(value) for key, value in best_config.to_dict().items()}


def create_plots(model_name):
    for coin in ["BTC", "ETH"]:
        for time_frame in timeframes:
            pred_plot(model_name, coin, time_frame)


def model_influential_plot(model_name):
    datasets = []
    for coin in all_coins:
        for time_frame in timeframes:
            datasets.append(get_analysis(model_name, coin, time_frame))

    influential_plot(datasets)


def coin_influential_plot(model_name, coin):
    datasets = []
    for time_frame in timeframes:
        datasets.append(get_analysis(model_name, coin, time_frame))
    influential_plot(datasets)


def time_frame_influential_plot(model_name, time_frame):
    datasets = []
    for coin in all_coins:
        datasets.append(get_analysis(model_name, coin, time_frame))
    influential_plot(datasets)


def influential_plot(datasets, correlation_type: str = "pearson"):
    # pearson, spearman, kendall

    # Initialize a dictionary to store the correlation coefficients
    correlations = {col: [] for col in datasets[0].columns if col != "rmse"}

    # Compute the correlation for each dataset
    for df in datasets:
        corr = df.corr(correlation_type)["rmse"]
        for col in correlations.keys():
            correlations[col].append(corr[col])

    # Compute the mean and standard deviation for each hyperparameter
    means = {col: np.mean(correlations[col]) for col in correlations.keys()}
    std_devs = {col: np.std(correlations[col]) for col in correlations.keys()}

    # Convert the results to pandas Series for easier plotting
    means_series = pd.Series(means)
    std_devs_series = pd.Series(std_devs)

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        means_series.index,
        means_series.sort_values(),
        yerr=std_devs_series,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
    )
    plt.axhline(y=0, color="r", linestyle="-")
    ax.set_ylabel("Mean Correlation with RMSE")
    ax.set_title("Hyperparameter Influence on RMSE")
    plt.show()


def model_analysis(model_name):
    # Get the best hyperparameters for each coin and time frame
    # Creating a overview of best hyperparameters, grouped by timeframe or coin
    # Also shows what hyperparameters are the most influential for the RMSE
    pass


def coin_analysis(model_name, coin):
    pass


def best_hyperparameters_model(model_name):
    all_best = []

    for coin in all_coins:
        for time_frame in timeframes:
            all_best.append(best_hyperparameters(model_name, coin, time_frame))

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_best)

    # Determine the number of rows and columns for the subplots
    n = len(df.columns)
    ncols = 3  # You can adjust this as needed
    nrows = n // ncols + (n % ncols > 0)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, nrows * 3)
    )  # Adjust the figure size as needed
    axes = axes.ravel()  # Flatten the axes array

    # For each column in the DataFrame (each hyperparameter), plot a bar plot in a subplot
    for i, col in enumerate(df.columns):
        if col == "dropout":
            counts, bins, patches = axes[i].hist(
                df[col],
                edgecolor="black",
            )  # Plot a histogram of the column
            # We'll color code by height, but you could use any scalar
            max_height = max(counts)
            # Loop over the bars
            for rect, count in zip(patches, counts):
                if count == max_height:
                    rect.set_facecolor("green")
                else:
                    rect.set_facecolor("blue")
        else:
            counts = df[col].value_counts().sort_index()
            colors = ["blue" if x < max(counts) else "green" for x in counts]
            counts.plot(
                kind="bar", ax=axes[i], width=1.0, edgecolor="black", color=colors
            )  # Plot a bar plot of the column
        axes[i].set_title(f"Distribution of {col}")  # Set the title of the subplot
        axes[i].set_xlabel(col)  # Set the x-label of the subplot
        axes[i].set_ylabel("Frequency")  # Set the y-label of the subplot

    # Remove any unused subplots
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust the layout so everything fits
    plt.show()  # Display the plot


def avg_best(model_name):
    datasets = []
    for coin in all_coins:
        for time_frame in timeframes:
            datasets.append(get_analysis(model_name, coin, time_frame))

    # Concatenate all results into a single dataframe
    all_results = pd.concat(datasets)

    columns = [
        "num_layers",
        "num_blocks",
        "layer_widths",
        "input_chunk_length",
        "n_epochs",
        "batch_size",
        # "dropout",
    ]

    # Group by the hyperparameters and calculate the mean RMSE for each group
    average_results = all_results.groupby(columns).mean()
    print(average_results)

    # Find the hyperparameters with the lowest average RMSE
    best_hyperparameters = average_results["rmse"].idxmin()

    # Make it a dictionary
    best_hyperparameters = dict(zip(columns, best_hyperparameters))

    print(f"Best hyperparameters on average: {best_hyperparameters}")


if __name__ == "__main__":
    model_name = "TCN"
    coin = "BNB"
    time_frame = "1m"

    # influential_parameters(model_name, coin, time_frame)
    # print(best_hyperparameters(model_name, coin, time_frame))
    # result_analysis(model_name, coin, time_frame)
    # model_influential_plot(model_name)
    best_hyperparameters_model(model_name)
    # coin_influential_plot(model_name, coin)
    # time_frame_influential_plot(model_name, time_frame)
    # avg_best(model_name)
