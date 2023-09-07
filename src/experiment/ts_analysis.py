import numpy as np
import plotly.graph_objects as go

import config
from experiment.utils import (
    all_model_predictions,
)
from experiment.rmse import read_rmse_csv


def compare_predictions(
    pred: str = config.log_returns_pred,
    coin: str = config.all_coins[0],
    time_frame: str = config.timeframes[-1],
):
    """
    Compare the predictions of all models for a given coin and time frame to their test data

    Parameters
    ----------
    model_dir : str
        One of the models declared in the config
    coin : str
        The coin to compare, e.g. "BTC", "ETH", "LTC"
    time_frame : str
        The time frame to compare, e.g. "1d", "4h", "15m"
    """

    # Get the predictions
    model_predictions, _ = all_model_predictions(pred, coin, time_frame)
    test = model_predictions[list(model_predictions.keys())[0]][1]

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
    for model_name, (model_pred, _, _) in model_predictions.items():
        # If the model is extended models, plot each prediction separately
        if pred == config.extended_model:
            for i, p in enumerate(model_pred):
                fig.add_trace(
                    go.Scatter(
                        y=p.univariate_values(),
                        mode="lines",
                        name=f"{model_name} {i}",
                        visible="legendonly",  # This line is hidden initially
                        legendgroup=model_name,
                        showlegend=True if i == 0 else False,
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    y=model_pred.univariate_values(),
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


def rmse_outliers_coin(pred: str, coin: str, time_frame: str):
    """
    Print the outliers for each model for a given coin and time frame

    Parameters
    ----------
    pred : str
        One of the predictions directories declared in the config
    coin : str
        The coin to compare, e.g. "BTC", "ETH", "LTC"
    time_frame : str
        The time frame to compare, e.g. "1d", "4h", "15m"
    """

    df = read_rmse_csv(pred, time_frame)

    # Only get the coin
    df = df.loc[coin]

    # Compute and print outliers for each model
    for pred, rmses in df.items():
        q1 = np.quantile(rmses, 0.25)
        q3 = np.quantile(rmses, 0.75)
        iqr = q3 - q1
        low_outliers = [
            (f"period: {i+1}", f"rmse: {x}")
            for i, x in enumerate(rmses)
            if x < (q1 - 1.5 * iqr)
        ]
        high_outliers = [
            (f"period: {i+1}", f"rmse: {x}")
            for i, x in enumerate(rmses)
            if x > (q3 + 1.5 * iqr)
        ]

        if low_outliers:
            print(f"Low outliers for {pred}: {low_outliers}")
        if high_outliers:
            print(f"High outliers for {pred}: {high_outliers}")


def all_models_outliers(pred: str, time_frame: str):
    """
    Prints all model outliers for each available coin for a given time frame

    Parameters
    ----------
    pred : str
        One of the preds declared in the config
    time_frame : str
        The time frame to compare, e.g. "1d", "4h", "15m"
    """

    df = read_rmse_csv(pred, time_frame)

    # Average the RMSEs
    df = df.applymap(lambda x: np.mean(x))

    q1 = df.quantile(0.25)

    q3 = df.quantile(0.75)

    IQR = q3 - q1

    low_outliers = df[df < (q1 - 1.5 * IQR)]
    high_outliers = df[df > (q3 + 1.5 * IQR)]

    # Remove rows with NaN
    low_outliers = low_outliers.dropna(how="all")
    high_outliers = high_outliers.dropna(how="all")

    print(low_outliers)
    print(high_outliers)


def compare_multiple_predictions(
    preds: list = config.log_preds,
    coin: str = "BTC",
    time_frame: str = "1d",
):
    """
    Compares the predictions of various models.

    Parameters
    ----------
    preds : list
        List of preds to compare, e.g. config.log_preds or config.raw_preds
    coin : str
        The coin to compare, e.g. "BTC", "ETH", "LTC"
    time_frame : str
        The time frame to compare, e.g. "1d", "4h", "15m"
    """

    # Create a new figure
    fig = go.Figure()

    for pred in preds:
        # Get the predictions
        model_predictions, _ = all_model_predictions(pred, coin, time_frame)

        # If this is the first model, add the test data line to the plot
        if pred == preds[0]:
            test = model_predictions[list(model_predictions.keys())[0]][1]
            fig.add_trace(
                go.Scatter(
                    y=test.univariate_values(),
                    mode="lines",
                    name="Test Set",
                    line=dict(color="black", width=2),
                )
            )

        # Plot each model's predictions
        for model_name, (model_pred, _, _) in model_predictions.items():
            fig.add_trace(
                go.Scatter(
                    y=model_pred.univariate_values(),
                    mode="lines",
                    name=f"({config.pred_names[pred]}) {model_name}",
                    visible="legendonly",  # This line is hidden initially
                    legendgroup=model_name,
                    showlegend=True if pred == preds[0] else False,
                    opacity=0.75,  # Setting opacity to 0.75
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
