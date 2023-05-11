import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries, concatenate
from darts.metrics import mape, mase, rmse


def plot_results(train, test, predictions, show_plots=True):
    errors_mape = []
    errors_mase = []
    errors_rmse = []

    preds_ts = []

    for i in range(len(predictions)):
        print("Period", i + 1)
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

    print("Average MAPE:", np.mean(errors_mape))
    print("Average MASE:", np.mean(errors_mase))
    print("Average RMSE:", np.mean(errors_rmse))

    all_preds = concatenate(preds_ts, axis=0)  # TimeSeries.stack(preds_ts)
    all_tests = concatenate(test, axis=0)  # TimeSeries.stack(test)

    # Plot the results
    if show_plots:
        plt.figure(figsize=(12, 6))
        plt.plot(all_tests.univariate_values(), label="Test Set")
        plt.plot(all_preds.univariate_values(), label="Forecast")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.title("Test Set vs. Forecast")
        plt.show()
