import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from experiment.rmse import read_rmse_csv
from experiment.utils import get_predictions


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

    model_1_rmse = read_rmse_csv(pred=model_1, time_frame=time_frame)
    model_2_rmse = read_rmse_csv(pred=model_2, time_frame=time_frame)

    if coin_as_x_axis:
        plot_items = config.all_models

        if model_1 in [config.extended_pred, config.extended_to_raw_pred]:
            plot_items = config.ml_models

        model_1_rmse = model_1_rmse.T
        model_2_rmse = model_2_rmse.T

    else:
        plot_items = config.all_coins

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


def plotly_model_boxplot(model: str = config.log_returns_pred, time_frame: str = "1d"):
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
    df = read_rmse_csv(pred=model, time_frame=time_frame)

    plotly_boxplot(
        df=df.T,
        plot_items=config.all_models,
        labels=config.all_coins,
    )


def plotly_coin_boxplot(model: str = config.log_returns_pred, time_frame: str = "1d"):
    """
    Plots a boxplot of the RMSEs for each model for the given coin.

    Parameters
    ----------
    model : str
        The model to plot, e.g. log_to_raw_pred.
    time_frame : str
        Time frame to plot, options are: "1m", "15m", "4h", "1d".
    """

    # For coin boxplot, call like this
    df = read_rmse_csv(pred=model, time_frame=time_frame)

    plotly_boxplot(
        df=df,
        plot_items=config.all_coins,
        labels=config.all_models,
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
    df = read_rmse_csv(pred=model, time_frame=time_frame)

    # Subset the dataframe
    if df_subset != "all models":
        df = df[df_subset]
    else:
        # Take the mean of the RMSEs for each model
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
    ax.set_xlabel("Forecasting Model")

    plt.show()


def plt_boxplots(
    dfs: list,
    models: list,
    outliers_percentile: int,
    y_min: int,
    use_hatches: bool = False,
    fontsize: int = 12,
    dark_mode: bool = True,
):
    """
     Plot a boxplot of the RMSEs for each DataFrame in dfs on the same plot.

    Parameters
    ----------
    dfs : list
        For instance a list of RMSE dataframes
    models : list
        The names of the models, column values in the dataframes
    outliers_percentile : int
        The percentile to use for the y-axis limits
    y_min : int
        The minimum value for the y-axis
    use_hatches : bool, optional
        If True gray scale values will be used, by default False
    fontsize : int, optional
        The size of the legend text and x and y-tick labels, by default 12
    """
    if dark_mode:
        plt.style.use("dark_background")
        colors = plt.cm.Dark2.colors
    else:
        colors = plt.cm.Accent.colors

    # Create a figure and axis
    _, ax = plt.subplots(figsize=(20, 8))

    if use_hatches:
        fill_styles = ["", "\\\\\\", "///"]
    else:
        fill_styles = colors

    legend_handles = []

    # Assuming all DataFrames have the same columns
    labels = dfs[0].columns.tolist()
    n_dfs = len(dfs)

    # Flatten all the data to calculate percentiles
    if type(outliers_percentile) == int:
        all_data = np.concatenate([df.values.flatten() for df in dfs])
        # Remove nan values in ndarray
        all_data = all_data[~np.isnan(all_data)]
        # Set the y-axis limits
        if y_min:
            ax.set_ylim(y_min, np.percentile(all_data, outliers_percentile))
        else:
            ax.set_ylim(np.min(all_data), np.percentile(all_data, outliers_percentile))

    # Width of each boxplot group
    x_positions = np.arange(len(labels))

    for i, (style, df) in enumerate(zip(fill_styles, dfs)):
        box_data = [df[col] for col in labels]

        # Offsetting positions for each DataFrame's boxplots
        positions = x_positions + (i - n_dfs / 2) * 0.2 + 0.1

        # Create the boxplot
        bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.1)

        for patch in bp["boxes"]:
            if use_hatches:
                patch.set(facecolor="white", hatch=style)
            else:
                patch.set(facecolor=style)

        # Create a custom legend handle
        legend_handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white" if use_hatches else style,
                edgecolor="black",
                hatch=style if use_hatches else None,
            )
        )

    # Calculate the average position for each group of boxplots
    average_positions = [
        np.mean([pos + (i - n_dfs / 2) * 0.2 + 0.2 for i in range(n_dfs)])
        for pos in x_positions
    ]

    # Set the x-ticks to the average positions
    ax.set_xticks(average_positions)

    # Adjust y-tick label font size
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fontsize)
    ax.legend(
        legend_handles,
        [config.pred_names[m] for m in models],
        loc="best",
        fontsize=fontsize,
    )
    # ax.set_title(f"The boxplots of RMSEs for each model on the {time_frame} time frame")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Forecasting Model")

    plt.tight_layout()
    plt.show()


def plt_single_df_boxplots(
    df,
    use_hatches: bool = False,
    outliers_percentile: int = 99,
    y_min: int = 0,
    x_label: str = "Cryptocurrency",
    y_label: str = "RMSE",
    rotate_xlabels: bool = False,
    fontsize: int = 12,
    first_white: bool = False,
    dark_mode: bool = True,
):
    """
    Plot a boxplot of the RMSEs for each item in a single DataFrame.

    Parameters
    ----------
    df: DataFrame
    use_hatches: bool, optional
        Whether to use hatches or distinct colors
    """
    if dark_mode:
        plt.style.use("dark_background")
        colors = plt.cm.Dark2.colors

        if first_white:
            # Add white in front
            colors = ((0.5, 0.5, 0.5),) + colors
    else:
        colors = plt.cm.Accent.colors

        if first_white:
            # Add white in front
            colors = ((1.0, 1.0, 1.0),) + colors

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(20, 8))

    hatches = ["", "\\\\\\", "///", "---", "|||"]  # Custom hatches

    # plt.cm.Paired.colors#plt.cm.viridis(np.linspace(0, 1, len(df.columns)))
    legend_handles = []

    # Flatten all the data to calculate percentiles
    if type(outliers_percentile) == int:
        all_data = np.concatenate(df.values.flatten())
        # Remove nan values in ndarray
        all_data = all_data[~np.isnan(all_data)]
        # Set the y-axis limits
        if y_min:
            ax.set_ylim(y_min, np.percentile(all_data, outliers_percentile))
        else:
            ax.set_ylim(np.min(all_data), np.percentile(all_data, outliers_percentile))

    # Assuming all DataFrames have the same columns
    labels = df.index.tolist()

    # Width of each boxplot group
    group_width = 0.8
    n_cols = len(df.columns)

    # Calculate the width of each individual boxplot within a group
    box_width = group_width / n_cols

    # Width of each boxplot group
    x_positions = np.arange(len(labels))

    # Initialize list to store the middle positions of each group of boxplots
    average_positions = []

    for i, column in enumerate(df.columns):
        box_data = [df.loc[row, column] for row in df.index]

        # Remove nan values in each sublist
        box_data = [[x for x in sublist if str(x) != "nan"] for sublist in box_data]

        # Offsetting positions for each DataFrame's boxplots
        offset = (i - n_cols / 2) * box_width + box_width / 2
        positions = x_positions + offset

        # Update average_positions
        if i == 0:
            average_positions = positions
        else:
            average_positions = [a + b for a, b in zip(average_positions, positions)]

        # Create the boxplot
        bp = ax.boxplot(
            box_data, positions=positions, patch_artist=True, widths=box_width
        )

        for patch in bp["boxes"]:
            if use_hatches:
                patch.set(facecolor="white", hatch=hatches[i])
            else:
                patch.set(facecolor=colors[i])

        # Create a custom legend handle
        legend_handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=colors[i] if not use_hatches else "white",
                edgecolor="black",
                hatch=hatches[i] if use_hatches else None,
            )
        )

    # Calculate the average position for each group of boxplots
    average_positions = [pos / n_cols for pos in average_positions]

    # Set the x-ticks to the average positions
    ax.set_xticks(average_positions)
    if rotate_xlabels:
        ax.set_xticklabels(labels, rotation=45, ha="center", fontsize=fontsize)
    else:
        ax.set_xticklabels(labels, ha="center", fontsize=fontsize)

    # Add legend
    ax.legend(legend_handles, list(df.columns), loc="best", fontsize=fontsize)

    # Add title and labels
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    ax.tick_params(axis="y", labelsize=fontsize)

    plt.tight_layout()
    plt.show()


def plt_multiple_df_boxplots(
    dfs: list,
    use_hatches: bool = False,
    outliers_percentile: list = [],
    y_min: int = 0,
    x_label: str = "Cryptocurrency",
    y_label: str = "RMSE",
    rotate_xlabels: bool = False,
    fontsize: int = 12,
    first_white: bool = False,
    dark_mode: bool = True,
):
    """
    Plot a boxplot of the RMSEs for each DataFrame in dfs on the same plot.

    Parameters
    ----------
    dfs: list of DataFrames
    use_hatches: bool, optional
        Whether to use hatches or distinct colors
    """
    # Create a figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(20, 8))  # Adjust figsize as needed

    if dark_mode:
        fig.set_facecolor("black")
    axs = axs.flatten()  # Flatten the array of axes for easier indexing

    hatches = ["", "\\\\\\", "///", "---", "|||"]  # Custom hatches

    if dark_mode:
        plt.style.use("dark_background")
        colors = plt.cm.Dark2.colors
        if first_white:
            # Add white in front
            colors = ((0.5, 0.5, 0.5),) + colors
    else:
        colors = plt.cm.Accent.colors
        if first_white:
            # Add white in front
            colors = ((1.0, 1.0, 1.0),) + colors

    if len(outliers_percentile) == 0:
        outliers_percentile = [100] * len(dfs)

    if len(outliers_percentile) != len(dfs):
        print("Length of outliers_percentile must be equal to length of dfs")
        return

    # Change if columns are > colors
    if len(dfs[0].columns) > len(colors):
        colors = plt.cm.tab20.colors

    legend_handles = []
    for ax_idx, df in enumerate(dfs):
        ax = axs[ax_idx]  # Select the current axis

        if dark_mode:
            ax.set_facecolor("black")
            ax.tick_params(axis="x", colors="white")  # X tick colors
            ax.tick_params(axis="y", colors="white")  # Y tick colors
            ax.spines["bottom"].set_color("white")  # X axis color
            ax.spines["left"].set_color("white")  # Y axis color
            ax.title.set_color("white")  # Title color
            ax.yaxis.label.set_color("white")  # Y axis label color
            ax.xaxis.label.set_color("white")  # X axis label color

        # adjust for percentile
        all_data = np.concatenate(df.values.flatten())
        all_data = all_data[~np.isnan(all_data)]
        if y_min:
            ax.set_ylim(y_min, np.percentile(all_data, outliers_percentile[ax_idx]))
        else:
            ax.set_ylim(
                np.min(all_data), np.percentile(all_data, outliers_percentile[ax_idx])
            )

        labels = df.index.tolist()
        group_width = 0.8
        n_cols = len(df.columns)
        box_width = group_width / n_cols
        x_positions = np.arange(len(labels))
        average_positions = []

        for i, column in enumerate(df.columns):
            box_data = [df.loc[row, column] for row in df.index]
            box_data = [[x for x in sublist if str(x) != "nan"] for sublist in box_data]
            offset = (i - n_cols / 2) * box_width + box_width / 2
            positions = x_positions + offset
            if i == 0:
                average_positions = positions
            else:
                average_positions = [
                    a + b for a, b in zip(average_positions, positions)
                ]
            bp = ax.boxplot(
                box_data, positions=positions, patch_artist=True, widths=box_width
            )
            for patch in bp["boxes"]:
                if use_hatches:
                    patch.set(facecolor="white", hatch=hatches[i])
                else:
                    patch.set(facecolor=colors[i])

        average_positions = [pos / n_cols for pos in average_positions]
        ax.set_xticks(average_positions)
        if rotate_xlabels:
            ax.set_xticklabels(labels, rotation=45, ha="center", fontsize=fontsize)
        else:
            ax.set_xticklabels(labels, ha="center", fontsize=fontsize)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.tick_params(axis="y", labelsize=fontsize)
        ax.set_title(config.tf_names[ax_idx])

    # Add legend for all subplots
    for i in range(n_cols):
        legend_handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=colors[i] if not use_hatches else "white",
                edgecolor="black",
                hatch=hatches[i] if use_hatches else None,
            )
        )

    # (left, bottom, right, top)
    plt.tight_layout(rect=[0.02, 0, 1, 0.90])

    # Create the legend for the entire figure
    fig.legend(
        legend_handles,
        list(dfs[0].columns),
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(legend_handles),
        fontsize=fontsize,
        title="Forecasting Model",
        title_fontsize=fontsize,
    )
    plt.show()


def plt_forecasting_models_comparison(
    pred: str = config.log_returns_pred,
    forecasting_models: list = ["ARIMA", "TBATS", "LSTM"],
    time_frame: list = "1d",
):
    # Read the RMSE data
    rmse_df = read_rmse_csv(pred, time_frame, avg=False, fill_NaN=True)
    rmse_df = rmse_df[forecasting_models]

    # Add scaled_to_log TCN
    if time_frame in ["15m", "1m"]:
        rmse_df2 = read_rmse_csv(
            config.scaled_to_log_pred, time_frame, avg=False, fill_NaN=True
        )

        if "RNN" in forecasting_models:
            rmse_df.drop(columns=["RNN"], inplace=True)
            rmse_df["RNN"] = rmse_df2["RNN"]
        if "TCN" in forecasting_models:
            rmse_df.drop(columns=["TCN"], inplace=True)
            rmse_df["TCN"] = rmse_df2["TCN"]

    if time_frame == "1d":
        outliers_percentile = 100
    if time_frame == "4h":
        outliers_percentile = 99
    elif time_frame == "15m":
        outliers_percentile = 97
    elif time_frame == "1m":
        outliers_percentile = 97

    plt_single_df_boxplots(rmse_df, outliers_percentile=outliers_percentile)


def plt_model_boxplot(model_dir: str, model: str, time_frame: str):
    plt_boxplot(model=model_dir, df_subset=model, time_frame=time_frame)


def plt_coin_boxplot(
    model_dir: str = config.log_returns_pred,
    coin: str = config.all_coins[0],
    time_frame: str = config.timeframes[-1],
):
    plt_boxplot(model=model_dir, df_subset=coin, time_frame=time_frame)


def all_models_boxplot(
    model: str = config.log_returns_pred, time_frame: str = config.timeframes[-1]
):
    plt_boxplot(model=model, df_subset="all models", time_frame=time_frame)


def complete_models_boxplot(
    preds: list = config.log_preds, time_frame: str = config.timeframes[-1]
):
    # Read the data
    if preds == config.log_preds:
        # Maybe change this depending on time frame
        outliers_percentile = 97
        if time_frame == "4h":
            outliers_percentile = 95
        if time_frame == "15m":
            outliers_percentile = 91
        if time_frame == "1m":
            outliers_percentile = 85
    else:
        outliers_percentile = 75

    # Read the RMSE data
    dfs = []
    for pred in preds:
        rmse_df = read_rmse_csv(pred, time_frame, avg=True, fill_NaN=True)
        dfs.append(rmse_df)

    plt_boxplots(
        dfs=dfs, models=preds, outliers_percentile=outliers_percentile, y_min=0
    )


def prediction_boxplots(
    pred: str = config.log_returns_pred,
    time_frame: str = config.timeframes[-1],
    coin: str = config.all_coins[0],
    models: list = config.all_models,
):
    # Initialize a dictionary to hold the combined data
    data_dict = {f"Period {i+1}": {} for i in range(config.n_periods)}

    # Loop through each model to get predictions and plot them
    for model_name in models:
        use_pred = None
        if time_frame in ["15m", "1m"]:
            if model_name in ["GRU", "TCN", "LSTM"]:
                use_pred = config.scaled_to_log_pred

        predictions, _, tests, _ = get_predictions(
            model=pred if use_pred is None else use_pred,
            forecasting_model=model_name,
            coin=coin,
            time_frame=time_frame,
        )

        actual_values = tests.pd_dataframe()
        forecast_values = predictions.pd_dataframe()

        period_size = len(actual_values) // config.n_periods
        forecast_splits = [
            forecast_values[i : i + period_size]["log returns"].tolist()
            for i in range(0, len(forecast_values), period_size)
        ]

        # Only do this once
        if model_name == models[0]:
            actual_splits = [
                actual_values[i : i + period_size]["log returns"].tolist()
                for i in range(0, len(actual_values), period_size)
            ]

            # Loop over each split, adding it to the data_dict
            for i, (actual_split) in enumerate(actual_splits):
                data_dict[f"Period {i+1}"][
                    "Test"
                ] = actual_split  # Replace actual_split with forecast_split if needed

        # Loop over each split, adding it to the data_dict
        for i, (forecast_split) in enumerate(forecast_splits):
            data_dict[f"Period {i+1}"][
                model_name
            ] = forecast_split  # Replace actual_split with forecast_split if needed

    # Convert the dictionary to a DataFrame
    df_combined = pd.DataFrame.from_dict(data_dict, orient="index")

    plt_single_df_boxplots(
        df_combined,
        outliers_percentile=100,
        x_label="Period",
        y_label="Logarithmic Returns",
        rotate_xlabels=False,
        first_white=True,
    )
