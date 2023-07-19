import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from data.csv_data import read_csv
from hyperopt.config import all_coins
from experiment.utils import all_model_predictions, read_rmse_csv


def compare_predictions(coin, time_frame):
    # Get the predictions
    model_predictions, _ = all_model_predictions(coin, time_frame)
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


def rmse_outliers_coin(coin, time_frame):
    df = read_rmse_csv(time_frame)

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


def all_models_outliers(time_frame):
    df = read_rmse_csv(time_frame)

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


def get_volatility_data(timeframe):
    complete_df = pd.DataFrame()

    for coin in all_coins:
        coin_df = read_csv(
            coin=coin, timeframe=timeframe, col_names=["volatility"]
        ).dropna()

        if complete_df.empty:
            complete_df.index = coin_df.index

        complete_df[coin] = coin_df["volatility"].tolist()

    return complete_df


def calculate_percentiles(complete_df):
    overall_median_volatility = complete_df.stack().median()
    overall_q3_volatility = complete_df.stack().quantile(0.75)
    overall_q1_volatility = complete_df.stack().quantile(0.25)

    return overall_median_volatility, overall_q3_volatility, overall_q1_volatility


def plot_lines(complete_df, ax):
    avg_volatility = complete_df.mean(axis=1)
    avg_line = plt.plot(
        avg_volatility,
        color="dodgerblue",
        linewidth=2.5,
        alpha=0.7,
        label="Average Volatility",
    )

    (
        overall_median_volatility,
        overall_q3_volatility,
        overall_q1_volatility,
    ) = calculate_percentiles(complete_df)

    overall_median_line = plt.axhline(
        y=overall_median_volatility,
        color="lime",
        linewidth=2,
        alpha=0.7,
        label="Overall Median Volatility",
    )
    overall_q3_line = plt.axhline(
        y=overall_q3_volatility,
        color="orange",
        linewidth=2,
        alpha=0.7,
        label="Overall 75th Percentile Volatility",
    )
    overall_q1_line = plt.axhline(
        y=overall_q1_volatility,
        color="darkred",
        linewidth=2,
        alpha=0.7,
        label="Overall 25th Percentile Volatility",
    )

    return avg_line, overall_median_line, overall_q3_line, overall_q1_line


def plot_train_test_periods(
    complete_df, ax, n_periods, test_size_percentage, val_size_percentage
):
    ts_length = 999
    test_size = int(ts_length / (1 / test_size_percentage - 1 + n_periods))
    train_size = int(test_size * (1 / test_size_percentage - 1))
    val_size = int(val_size_percentage * train_size)
    train_size = train_size - val_size

    _, ymax = ax.get_ylim()

    line_start = ymax * 2
    training_lines = []
    validation_lines = []
    testing_lines = []
    for i in range(n_periods):
        train_start = i * test_size
        train_end = train_start + train_size

        date_min = complete_df.index.min()
        date_max = complete_df.index.max()

        train_line_start = (complete_df.index[train_start] - date_min) / (
            date_max - date_min
        )
        train_line_end = (complete_df.index[train_end] - date_min) / (
            date_max - date_min
        )
        val_end = (complete_df.index[train_end + val_size] - date_min) / (
            date_max - date_min
        )
        test_end = (complete_df.index[min(train_end + test_size, 969)] - date_min) / (
            date_max - date_min
        )

        training_lines.append(
            plt.axhline(
                y=line_start,
                xmin=train_line_start,
                xmax=train_line_end,
                color="blue",
                linewidth=4,
                label="Training Periods",
            )
        )
        validation_lines.append(
            plt.axhline(
                y=line_start,
                xmin=train_line_end,
                xmax=val_end,
                color="green",
                linewidth=4,
                label="Validation Periods",
            )
        )
        testing_lines.append(
            plt.axhline(
                y=line_start,
                xmin=val_end,
                xmax=test_end,
                color="red",
                linewidth=4,
                label="Test Periods",
            )
        )
        line_start -= ymax * 0.1

    return training_lines, validation_lines, testing_lines


def plot_volatility(
    timeframe="1d", n_periods=5, test_size_percentage=0.25, val_size_percentage=0.1
):
    complete_df = get_volatility_data(timeframe)
    ax = complete_df.plot(figsize=(12, 6), alpha=0.2, color="grey", legend=False)

    avg_line, overall_median_line, overall_q3_line, overall_q1_line = plot_lines(
        complete_df, ax
    )
    training_lines, validation_lines, testing_lines = plot_train_test_periods(
        complete_df, ax, n_periods, test_size_percentage, val_size_percentage
    )

    # Create first legend
    first_legend = ax.legend(
        handles=[avg_line[0], overall_median_line, overall_q3_line, overall_q1_line],
        loc="best",
    )

    # Add the first legend manually to the current Axes.
    ax.add_artist(first_legend)

    # Create second legend
    ax.legend(
        handles=[training_lines[0], validation_lines[0], testing_lines[0]],
        loc="upper center",
        ncols=3,
        bbox_to_anchor=(0.5, 1.05),
    )

    ax.set_ylabel("Volatility")
    ax.set_xlabel("Date")

    plt.show()
