import numpy as np
import plotly.graph_objects as go

from experiment.utils import (
    all_model_predictions,
    read_rmse_csv,
)


def compare_predictions(model_dir: str, coin: str, time_frame: str):
    # Get the predictions
    model_predictions, _ = all_model_predictions(model_dir, coin, time_frame)
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
    for model_name, (pred, _, _) in model_predictions.items():
        if model_dir == "extended_models":
            for i, p in enumerate(pred):
                fig.add_trace(
                    go.Scatter(
                        y=p.univariate_values(),
                        mode="lines",
                        name=f"{model_name} {i}",
                        visible="legendonly",  # This line is hidden initially
                    )
                )
        else:
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


def rmse_outliers_coin(model_dir: str, coin: str, time_frame: str):
    df = read_rmse_csv(model_dir, time_frame)

    # Only get the coin
    df = df.loc[coin]

    # Compute and print outliers for each model
    for model, rmses in df.items():
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
            print(f"Low outliers for {model}: {low_outliers}")
        if high_outliers:
            print(f"High outliers for {model}: {high_outliers}")


def all_models_outliers(model_dir: str, time_frame: str):
    df = read_rmse_csv(model_dir, time_frame)

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
