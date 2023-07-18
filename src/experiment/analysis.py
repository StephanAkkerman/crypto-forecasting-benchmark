import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hyperopt.search_space import model_config
from hyperopt.config import all_coins
from experiment.utils import all_model_predictions, read_rmse_csv


def compare_predictions(coin, time_frame):
    # Get the predictions
    model_predictions, rmse_df = all_model_predictions(coin, time_frame)
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


def plotly_model_boxplot(time_frame, show_points: bool = False):
    """
    Interactively plot a boxplot of the RMSEs for each model.

    Parameters
    ----------
    time_frame : str
        Options are: "1m", "15m", "4h", "1d".
    """

    # Set the boxpoints
    boxpoints = "outliers"
    if show_points:
        boxpoints = "all"

    df = read_rmse_csv(time_frame)

    # Define your list of models
    models = list(model_config) + ["ARIMA", "TBATS"]

    # Create figure with secondary y-axis
    fig = make_subplots()

    # Create a dropdown menu
    buttons = []
    
    # Calculate total number of traces
    total_traces = len(all_coins) * len(df.columns)

    # Add traces, one for each model
    for i, model in enumerate(models):
        # Add a box trace for the current model
        for coin, rmse in df[model].items():
            fig.add_trace(go.Box(y=rmse, name=coin, boxpoints=boxpoints, visible=i==0))
            
        # Create a visibility list for current coin
        visibility = [False] * total_traces
        visibility[i * len(df.columns) : (i + 1) * len(df.columns)] = [True] * len(
            df.columns
        )

        # Add a button for the current model
        button = dict(
            label=model,
            method="update",
            args=[
                {"visible": [i == j for j in range(len(models))]},
                {"title": f"Boxplot for {model}"},
            ],
        )
        buttons.append(button)

    # Add dropdown menu to layout
    fig.update_layout(updatemenus=[go.layout.Updatemenu(active=0, buttons=buttons)])

    # Set title
    fig.update_layout(title_text=f"Boxplot for {models[0]}")

    # Use the same x-axis range for all traces
    fig.update_xaxes(categoryorder="array", categoryarray=all_coins)

    fig.show()


def plt_model_boxplot(model, time_frame):
    # x axis should show coins
    # y axis should show rmse boxplot values

    df = read_rmse_csv(time_frame)
    df = df[model]

    print(df)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create boxplot for each model
    boxplots = []
    labels = []
    for model, rmses in df.items():
        # Append boxplot data and labels
        boxplots.append(rmses)
        labels.append(model)

    # Create the boxplot with labels
    ax.boxplot(boxplots, labels=labels)

    # Set the labels
    ax.set_title("Boxplot of RMSEs")
    ax.set_ylabel("RMSE")

    plt.show()


def plotly_coin_boxplot(time_frame):
    df = read_rmse_csv(time_frame)

    # Create figure with secondary y-axis
    fig = make_subplots()

    buttons = []

    # Calculate total number of traces
    total_traces = len(all_coins) * len(df.columns)

    # Add traces, one for each model
    for i, coin in enumerate(all_coins):
        # Add a box trace for the current model
        for model, rmse in df.loc[coin].items():
            fig.add_trace(go.Box(y=rmse, name=model, visible=i == 0))

        # Create a visibility list for current coin
        visibility = [False] * total_traces
        visibility[i * len(df.columns) : (i + 1) * len(df.columns)] = [True] * len(
            df.columns
        )

        # Create a dropdown menu
        button = dict(
            label=coin,
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"Boxplot for {coin}"},
            ],
        )
        buttons.append(button)

    # Add dropdown menu to layout
    fig.update_layout(updatemenus=[go.layout.Updatemenu(active=0, buttons=buttons)])

    # Set title
    fig.update_layout(title_text=f"Boxplot for {all_coins[0]}")

    # Use the same x-axis range for all traces
    fig.update_xaxes(
        categoryorder="array", categoryarray=list(model_config) + ["ARIMA", "TBATS"]
    )

    fig.show()


def plt_coin_boxplot(coin, time_frame):
    df = read_rmse_csv(time_frame)

    # Only get the coin
    df = df.loc[coin]

    # Create a figure and axis
    _, ax = plt.subplots(figsize=(15, 6))

    # Create boxplot for each model
    boxplots = []
    labels = []
    for model, rmses in df.items():
        # Append boxplot data and labels
        boxplots.append(rmses)
        labels.append(model)

    # Create the boxplot with labels
    ax.boxplot(boxplots, labels=labels)

    # Set the labels
    ax.set_title("Boxplot of RMSEs")
    ax.set_ylabel("RMSE")

    plt.show()


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


def all_models_boxplot(time_frame):
    df = read_rmse_csv(time_frame)

    # Average the RMSEs
    df = df.applymap(lambda x: np.mean(x))

    # Create a figure and axis
    _, ax = plt.subplots(figsize=(15, 6))

    # Create boxplot for each model
    boxplots = []
    labels = []
    for model, rmses in df.items():
        # Append boxplot data and labels
        boxplots.append(rmses)
        labels.append(model)

    # Create the boxplot with labels
    ax.boxplot(boxplots, labels=labels)

    # Set the labels
    ax.set_title("Boxplot of RMSEs")
    ax.set_ylabel("RMSE")

    plt.show()


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
