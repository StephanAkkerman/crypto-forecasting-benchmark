import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiment.rmse import read_rmse_csv, extended_rmse_df
import config
from config import all_models, all_coins, raw_model, log_to_raw_model


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
    df : pd.DataFrame
        DataFrame with the RMSEs for each model and item.
    plot_items : list
        List of items to plot boxplots for, often the names of the coins. The values displayed on the x-axis.
        Should be the same as the index values.
    labels : list
        List of labels for each boxplot, often the names of the models. The values displayed in the selection box.
        Should be the same as the column names.
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


def plotly_boxplot_comparison(
    model_1: str,
    model_2: str,
    time_frame: str = "1d",
    coin_as_x_axis: bool = True,
    labels: list = None,
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

    model_1_rmse = read_rmse_csv(model=model_1, time_frame=time_frame)
    model_2_rmse = read_rmse_csv(model=model_2, time_frame=time_frame)

    if coin_as_x_axis:
        plot_items = all_models

        if model_1 in [config.extended_model, config.extended_to_raw_model]:
            plot_items = config.ml_models

        model_1_rmse = model_1_rmse.T
        model_2_rmse = model_2_rmse.T

    else:
        plot_items = all_coins

    # Set the boxpoints
    boxpoints = "outliers"
    if show_points:
        boxpoints = "all"

    # Create figure with secondary y-axis
    fig = make_subplots()

    # Create a dropdown menu
    buttons = []

    # Add traces, one for each item
    for i, item in enumerate(plot_items):
        # Add a box trace for the current item
        for (label, rmse_1), (_, rmse_2) in zip(
            model_1_rmse.loc[item].items(), model_2_rmse.loc[item].items()
        ):
            fig.add_trace(
                go.Box(
                    y=rmse_1,
                    name=f"{model_1} {label}",
                    boxpoints=boxpoints,
                    visible=i == 0,
                    legendgroup=label,
                )
            )
            fig.add_trace(
                go.Box(
                    y=rmse_2,
                    name=f"{model_2} {label}",
                    boxpoints=boxpoints,
                    visible=i == 0,
                    legendgroup=label,
                    showlegend=False,
                )
            )

        # Create a visibility list for current item
        visibility = [False] * len(plot_items) * len(model_1_rmse.columns) * 2
        visibility[
            i * len(model_1_rmse.columns) * 2 : (i + 1) * len(model_1_rmse.columns) * 2
        ] = ([True] * len(model_1_rmse.columns) * 2)

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
    if labels:
        fig.update_xaxes(categoryorder="array", categoryarray=labels)

    fig.show()


def plotly_model_boxplot(model: str = log_to_raw_model, time_frame: str = "1d"):
    """
    Plots a boxplot of the RMSEs for each coin for the given model.

    Parameters
    ----------
    model : str
        The model to plot, e.g. transformed_model.
    time_frame : str
        Time frame to plot, options are: "1m", "15m", "4h", "1d".
    """

    # For model boxplot, call like this
    df = read_rmse_csv(model_di=model, time_frame=time_frame)

    plotly_boxplot(
        df=df.T,
        plot_items=all_models,
        labels=all_coins,
    )


def plotly_coin_boxplot(model: str = log_to_raw_model, time_frame: str = "1d"):
    """
    Plots a boxplot of the RMSEs for each model for the given coin.

    Parameters
    ----------
    model : str
        The model to plot, e.g. log_to_raw_model.
    time_frame : str
        Time frame to plot, options are: "1m", "15m", "4h", "1d".
    """

    # For coin boxplot, call like this
    df = read_rmse_csv(model=model, time_frame=time_frame)

    plotly_boxplot(
        df=df,
        plot_items=all_coins,
        labels=all_models,
    )


def plt_boxplot(
    model: str,
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
    df = read_rmse_csv(model=model, time_frame=time_frame)

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
    plt_boxplot(model=model_dir, df_subset=model, time_frame=time_frame)


def plt_coin_boxplot(model_dir, coin, time_frame):
    plt_boxplot(model=model_dir, df_subset=coin, time_frame=time_frame)


def all_models_boxplot(model_dir, time_frame):
    plt_boxplot(model=model_dir, df_subset="all models", time_frame=time_frame)


def plotly_extended_model_rmse(time_frame):
    df = extended_rmse_df(time_frame=time_frame)

    # Change index
    labels = [f"Until Period {i}" for i in df.index.tolist()]
    df.index = labels

    plotly_boxplot(df=df.T, labels=labels, plot_items=config.ml_models)


def plt_extended_model_rmse(time_frame: str):
    # Get RMSE data
    rmse_df = read_rmse_csv(model=config.extended_model, time_frame=time_frame)

    data = {}
    for model in rmse_df.columns:
        # Get the RMSEs for the given model
        data[model] = rmse_df[model].iloc[: config.n_periods].tolist()

    n_models = len(data)
    n_periods = len(next(iter(data.values())))

    # Generate a list of unique colors for each model
    colors = plt.cm.viridis(np.linspace(0, 1, n_models))

    fig, ax = plt.subplots(figsize=(12, 8))

    # For each period
    for period in range(n_periods):
        # For each model
        for model_idx, model_name in enumerate(data):
            # Plot the boxplot for this model and this period with the model's unique color
            box = ax.boxplot(
                data[model_name][period],
                positions=[period * n_models + model_idx],
                widths=0.6,
                patch_artist=True,
            )
            for patch in box["boxes"]:
                patch.set_facecolor(colors[model_idx])

    # Set the x-ticks for each model within each period
    ax.set_xticks([i for i in range(n_models * n_periods)])

    # Set the x-tick labels to be the model names, repeated for each period
    ax.set_xticklabels(list(data.keys()) * n_periods, rotation=45, ha="right")

    ax.set_title("Comparison of RMSE values across models and periods")
    ax.set_xlabel("Period and Model")
    ax.set_ylabel("RMSE")

    # Create a custom legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=4, label=model_name)
        for i, model_name in enumerate(data)
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()
