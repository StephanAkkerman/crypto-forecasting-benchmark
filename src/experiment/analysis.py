import os
import pandas as pd
from darts.timeseries import TimeSeries
from darts import concatenate
from darts.metrics import rmse
import plotly.graph_objects as go
from hyperopt.search_space import model_config


def get_predictions(model_name, coin, time_frame):
    preds = []
    tests = []
    rmses = []

    for period in range(5):
        file_path = f"data/models/{model_name}/{coin}/{time_frame}/pred_{period}.csv"
        if not os.path.exists(file_path):
            print(
                f"Skipping {model_name} as it does not exist in the data/models/ directory."
            )
            return None, None, None

        pred = pd.read_csv(file_path)
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
        preds, tests, rmses = get_predictions(model, coin, time_frame)
        if preds is not None:
            model_predictions[model] = (preds, tests, rmses)

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
