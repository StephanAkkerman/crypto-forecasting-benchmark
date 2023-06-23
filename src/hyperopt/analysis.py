import os
import glob
import matplotlib.pyplot as plt
import pandas as pd


def save_plot(save_loc: str):
    """
    Create a plot of the forecast and save it to save_loc.

    Parameters
    ----------
    save_loc : str
        The location to save the plot to.
    """

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
            rmse = file.split("_")[0]
            rmse = rmse[rmse.find("0.") :]
            print(rmse)
            predictions.append((pd.read_csv(file), rmse))

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    # Loop over items in dict
    for df, rmse in predictions:
        df["log returns"].plot(ax=ax, label=f"RMSE: {rmse}")
    # plt.plot(val_data, label="Test Set")
    val_data["log returns"].plot(ax=ax, label="Test Set")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Test Set vs. Forecast")
    plt.savefig(f"{save_loc}/forecast.png")
    plt.close()


if __name__ == "__main__":
    model_name = "NBEATS"
    coin = "BTC"
    time_frame = "1d"

    save_loc = f"hyperopt_results/{model_name}/{coin}/{time_frame}/"

    save_plot(save_loc)
