import pandas as pd
import matplotlib.pyplot as plt
from darts import concatenate
from darts.metrics import mae, mase, rmse


def eval_model(model_name, coin, time_frame, train, test, predictions, show_plots=True):
    errors_mase = []
    errors_rmse = []
    errors_mae = []

    for i in range(len(predictions)):
        # Calculate the mean squared error
        errors_mase.append(mase(test[i], predictions[i], train[i]))
        errors_rmse.append(rmse(test[i], predictions[i]))
        errors_mae.append(mae(test[i], predictions[i]))

    results = pd.DataFrame(
        {"MAE": errors_mae, "MASE": errors_mase, "RMSE": errors_rmse}
    )

    print(results)

    csv_file_loc = f"data/models/{model_name}/{coin}/{time_frame}_metrics.csv"
    plot_file_loc = f"data/models/{model_name}/{coin}/plots/{time_frame}.png"
    forecast_loc = f"data/models/{model_name}/{coin}/{time_frame}_forecast.csv"

    make_plot(
        results,
        csv_file_loc,
        plot_file_loc,
        forecast_loc,
        predictions,
        test,
        show_plots,
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
