import os
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries, concatenate
from darts.metrics import mape, mase, rmse


def eval_model(model_name, coin, time_frame, train, test, predictions, show_plots=True):
    errors_mape = []
    errors_mase = []
    errors_rmse = []

    preds_ts = []

    for i in range(len(predictions)):
        preds = pd.DataFrame({"preds": predictions[i], "date": test[i].time_index})
        preds = TimeSeries.from_dataframe(preds, "date", "preds")
        preds_ts.append(preds)

        # Calculate the mean squared error
        errors_mape.append(mape(test[i], preds))
        errors_mase.append(mase(test[i], preds, train[i]))
        errors_rmse.append(rmse(test[i], preds))

    results = pd.DataFrame(
        {"MAPE": errors_mape, "MASE": errors_mase, "RMSE": errors_rmse}
    )

    print(results)

    # Add the folders if they don't exist
    if not os.path.exists(f"data/models/{model_name}/{coin}"):
        if not os.path.exists(f"data/models/{model_name}"):
            if not os.path.exists("data/models"):
                os.makedirs("data/models")
            os.makedirs(f"data/models/{model_name}")
        os.makedirs(f"data/models/{model_name}/{coin}")
        os.makedirs(f"data/models/{model_name}/{coin}/plots")

    csv_file_loc = f"data/models/{model_name}/{coin}/{time_frame}_metrics.csv"
    plot_file_loc = f"data/models/{model_name}/{coin}/plots/{time_frame}.png"
    forecast_loc = f"data/models/{model_name}/{coin}/{time_frame}_forecast.csv"

    make_plot(
        results, csv_file_loc, plot_file_loc, forecast_loc, preds_ts, test, show_plots
    )


def make_plot(
    results, csv_file_loc, plot_file_loc, forecast_loc, preds_ts, test, show_plots
):  # Save the results to a CSV file
    results.to_csv(csv_file_loc, index=False)

    all_preds = concatenate(preds_ts, axis=0)
    all_tests = concatenate(test, axis=0)

    all_preds.to_csv(forecast_loc, index=False)

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(all_tests.univariate_values(), label="Test Set")
    plt.plot(all_preds.univariate_values(), label="Forecast")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Test Set vs. Forecast")
    plt.savefig(plot_file_loc)
    if show_plots:
        plt.show()
    plt.close()
