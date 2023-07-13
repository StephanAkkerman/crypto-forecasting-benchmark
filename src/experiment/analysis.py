import pandas as pd
import matplotlib.pyplot as plt
from darts.timeseries import TimeSeries
from darts import concatenate
from darts.metrics import mae, mase, rmse
import plotly.graph_objects as go
from hyperopt.search_space import model_config


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


def get_predictions(model_name, coin, time_frame):
    preds = []
    tests = []
    rmses = []

    for period in range(5):
        pred = pd.read_csv(
            f"data/models/{model_name}/{coin}/{time_frame}/pred_{period}.csv"
        )
        pred = TimeSeries.from_dataframe(
            pred, time_col="time", value_cols=["log returns"]
        )

        test = pd.read_csv(
            f"data/models/{model_name}/{coin}/{time_frame}/test_{period}.csv"
        )
        test = TimeSeries.from_dataframe(
            test, time_col="date", value_cols=["log returns"]
        )
        rmses.append(rmse(test, pred))

        # Add it to list
        preds.append(pred)
        tests.append(test)

    preds = concatenate(preds, axis=0)
    tests = concatenate(tests, axis=0)

    return preds, tests, rmses


def all_model_predictions(coin, time_frame):
    model_predictions = {}

    models = list(model_config) + ["ARIMA"]

    for model in models:
        model_predictions[model] = get_predictions(model, coin, time_frame)

    # Only use the third value in the tuple (the rmse)
    rmses = {model: rmse for model, (_, _, rmse) in model_predictions.items()}
    rmse_df = pd.DataFrame(rmses)

    # Add average row to dataframe
    rmse_df.loc["Average"] = rmse_df.mean()

    print(rmse_df)

    return model_predictions


def compare_predictions(coin, time_frame):
    # Get the predictions
    model_predictions = all_model_predictions(coin, time_frame)
    test = model_predictions["ARIMA"][1]

    # Create a new figure
    fig = go.Figure()

    # Add test data line
    fig.add_trace(
        go.Scatter(
            y=test.univariate_values(),
            mode="lines",
            name="Test Set",
            line=dict(color="black", width=2),
        )
    )

    # Plot each model's predictions
    for model_name, (pred, _, _) in model_predictions.items():
        fig.add_trace(
            go.Scatter(
                y=pred.univariate_values(),
                mode="lines",
                name=model_name,
                visible="legendonly",  # This line is hidden initially
            )
        )

    # Compute the length of each period
    period_length = len(test) // 5

    # Add a vertical line at the end of each period
    for i in range(1, 5):
        fig.add_shape(
            type="line",
            x0=i * period_length,
            y0=0,
            x1=i * period_length,
            y1=1,
            yref="paper",
            line=dict(color="Red", width=1.5),
        )

    # Add labels and title
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Value",
        title=f"{coin} Predictions Comparison",
    )

    # Show the plot
    fig.show()
