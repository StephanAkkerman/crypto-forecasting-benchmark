import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from hyperopt.search_space import model_config
from hyperopt.config import all_coins
from experiment.utils import read_rmse_csv


def plotly_boxplot(
    df: pd.DataFrame,
    plot_items: list,
    labels: list,
    title_prefix: str = "Boxplot",
    show_points: bool = False,
):
    """
    Interactively plot a boxplot of the RMSEs for each item in plot_items.

    Parameters
    ----------
    model_dir: str
        Directory where the models are saved.
    time_frame : str
        Options are: "1m", "15m", "4h", "1d".
    plot_items : list
        List of items to plot boxplots for, often the names of the coins.
    labels : list
        List of labels for each boxplot, often the names of the models.
    title_prefix : str
        Prefix for the plot title.
    """

    # Set the boxpoints
    boxpoints = "outliers"
    if show_points:
        boxpoints = "all"

    # Create figure with secondary y-axis
    fig = make_subplots()

    # Create a dropdown menu
    buttons = []

    # Calculate total number of traces
    total_traces = len(plot_items) * len(df.columns)

    # Add traces, one for each item
    for i, item in enumerate(plot_items):
        # Add a box trace for the current item
        for label, rmse in df.loc[item].items():
            fig.add_trace(
                go.Box(y=rmse, name=label, boxpoints=boxpoints, visible=i == 0)
            )

        # Create a visibility list for current item
        visibility = [False] * total_traces
        visibility[i * len(df.columns) : (i + 1) * len(df.columns)] = [True] * len(
            df.columns
        )

        # Add a button for the current item
        button = dict(
            label=item,
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"{title_prefix} for {item}"},
            ],
        )
        buttons.append(button)

    # Add dropdown menu to layout
    fig.update_layout(updatemenus=[go.layout.Updatemenu(active=0, buttons=buttons)])

    # Set title
    fig.update_layout(title_text=f"{title_prefix} for {plot_items[0]}")

    # Use the same x-axis range for all traces
    fig.update_xaxes(categoryorder="array", categoryarray=labels)

    fig.show()


def plotly_model_boxplot(model_dir: str, time_frame: str):
    # For model boxplot, call like this
    df = read_rmse_csv(model_dir=model_dir, time_frame=time_frame)

    plotly_boxplot(
        df=df.T,
        plot_items=list(model_config) + ["ARIMA", "TBATS"],
        labels=all_coins,
    )


def plotly_coin_boxplot(model_dir: str, time_frame: str):
    # For coin boxplot, call like this
    df = read_rmse_csv(model_dir=model_dir, time_frame=time_frame)

    plotly_boxplot(
        df=df,
        plot_items=all_coins,
        labels=list(model_config) + ["ARIMA", "TBATS"],
    )


def plt_boxplot(
    model_dir: str,
    df_subset: str,
    title: str = "Boxplot of RMSEs",
    time_frame: str = None,
):
    """
    Plot a boxplot of the RMSEs.

    Parameters
    ----------
    model_dir: str
        Directory where the models are saved.
    df_subset : str
        Either 'model' or 'coin'. Determines whether to subset dataframe by model or coin.
    title : str
        Title of the plot.
    average : bool, optional
        Whether to average the RMSEs. The default is False.
    time_frame : str, optional
        Time frame to subset dataframe. Only needed if df_subset is 'model'.
    """

    # Read in the dataframe
    df = read_rmse_csv(model_dir=model_dir, time_frame=time_frame)

    # Subset the dataframe
    if df_subset != "all models":
        df = df[df_subset]
    else:
        df = df.applymap(lambda x: np.mean(x))

    # Create a figure and axis
    _, ax = plt.subplots(figsize=(15, 6))

    # Create boxplot for each model
    boxplots = []
    labels = []
    for item, rmses in df.items():
        # Append boxplot data and labels
        boxplots.append(rmses)
        labels.append(item)

    # Create the boxplot with labels
    ax.boxplot(boxplots, labels=labels)

    # Set the labels
    ax.set_title(title)
    ax.set_ylabel("RMSE")

    plt.show()


def plt_model_boxplot(model_dir: str, model: str, time_frame: str):
    plt_boxplot(model_dir=model_dir, df_subset=model, time_frame=time_frame)


def plt_coin_boxplot(model_dir, coin, time_frame):
    plt_boxplot(model_dir=model_dir, df_subset=coin, time_frame=time_frame)


def all_models_boxplot(model_dir, time_frame):
    plt_boxplot(model_dir=model_dir, df_subset="all models", time_frame=time_frame)
