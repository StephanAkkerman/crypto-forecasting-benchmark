import os
import glob
import matplotlib.pyplot as plt
import pandas as pd

from config import timeframes


def save_plot(model_name: str, coin: str, time_frame: str, save: bool = True):
    """
    Create a plot of the forecast and save it to save_loc.

    Parameters
    ----------
    save_loc : str
        The location to save the plot to.
    """

    save_loc = f"hyperopt_results/{model_name}/{coin}/{time_frame}/"

    # Get all .csv files in save_loc
    csv_files = glob.glob(os.path.join(save_loc, "*.csv"))

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
        plt.savefig(f"{save_loc}/forecast.png")
    else:
        plt.show()
    plt.close()


def best_hyperparameters(model_name, coin, time_frame):
    # Read the results
    save_loc = f"hyperopt_results/{model_name}/{coin}/{time_frame}/"

    # Get the period0_results.csv
    results = pd.read_csv(f"{save_loc}period0_results.csv")

    # Only keep useful columns
    results = results.drop(
        [
            "time_this_iter_s",
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
            "config/model_name",
            "config/pl_trainer_kwargs/enable_progress_bar",
            "config/pl_trainer_kwargs/accelerator",
        ],
        axis=1,
    )

    # Rename columns starting with config/
    results.rename(columns=lambda x: x.replace("config/", ""), inplace=True)

    # Sort by rmse
    results = results.sort_values(by=["rmse"])

    print(results)

    # Save the results
    # results.to_csv(f"{save_loc}best_hyperparameters.csv", index=False)


def create_plots(model_name):
    for coin in ["BTC", "ETH"]:
        for time_frame in timeframes:
            save_plot(model_name, coin, time_frame)


def result_analysis(model_name, coin, time_frame):
    best_hyperparameters(model_name, coin, time_frame)
    save_plot(model_name, coin, time_frame, save=False)


def model_analysis(model_name):
    # Get the best hyperparameters for each coin and time frame
    # Creating a overview of best hyperparameters, grouped by timeframe or coin
    pass


if __name__ == "__main__":
    model_name = "NBEATS"
    coin = "BNB"
    time_frame = "1m"

    result_analysis(model_name, coin, time_frame)
