import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Local imports
import config
from experiment.utils import all_model_predictions


def build_complete_rmse_database(skip_existing: bool = True):
    """Build the RMSE database for all models and time frames."""
    models = [
        config.log_returns_pred,
        config.log_to_raw_pred,
        config.raw_pred,
        config.raw_to_log_pred,
        config.scaled_pred,
        config.scaled_to_log_pred,
        config.scaled_to_raw_pred,
        config.extended_pred,
        config.extended_to_raw_pred,
    ]

    for model in models:
        build_rmse_database(pred=model, skip_existing=skip_existing)


def read_rmse_csv(
    pred: str,
    time_frame: str,
    avg: bool = False,
    add_mcap: bool = False,
    ignore_model=[],
    fill_NaN: bool = True,
) -> pd.DataFrame:
    # Read the data from the .csv
    df = pd.read_csv(
        f"{config.rmse_dir}/{pred}/rmse_{time_frame}.csv", index_col=0
    ).drop(columns=ignore_model)

    # Convert string to list of floats
    df = df.applymap(lambda x: x.strip("[]").split(", "))

    # Convert list of strings to list of floats
    df = df.applymap(lambda x: [float(i) for i in x])

    if fill_NaN:
        # Function to fill NaN values in a list with the non-NaN value in the list
        def fill_list_nan(lst):
            fill_value = next((x for x in lst if not np.isnan(x)), np.nan)
            return [fill_value if np.isnan(x) else x for x in lst]

        # Apply the fill_list_nan function to each cell in the DataFrame
        df = df.applymap(fill_list_nan)

    # Round the values in the list
    if avg:
        df = df.applymap(lambda x: np.mean(x))

    if add_mcap:
        df["mcap category"] = df.index.map(assign_mcap_category)

    return df


# Function to assign market capitalization category based on index
def assign_mcap_category(crypto_index):
    if crypto_index in config.large_cap:
        return "Large"
    elif crypto_index in config.mid_cap:
        return "Mid"
    elif crypto_index in config.small_cap:
        return "Small"

    raise ValueError(f"Invalid crypto index: {crypto_index}")


def build_rmse_database(
    pred: str = config.log_returns_pred, skip_existing: bool = True
):
    os.makedirs(f"{config.rmse_dir}/{pred}", exist_ok=True)

    for tf in config.timeframes:
        # Skip if the file already exists
        if skip_existing:
            if os.path.exists(f"{config.rmse_dir}/{pred}/rmse_{tf}.csv"):
                print(
                    f"{config.rmse_dir}/{pred}/rmse_{tf}.csv already exists, skipping..."
                )
                continue

        print(f"Building {config.rmse_dir}/{pred}/rmse_{tf}.csv...")

        # Data will be added to this DataFrame
        rmse_df = pd.DataFrame()

        for coin in config.all_coins:
            # Get the predictions
            _, rmse_df_coin = all_model_predictions(
                model=pred, coin=coin, time_frame=tf
            )
            # Convert the dataframe to a list of lists
            rmse_df_list = pd.DataFrame(
                {col: [rmse_df_coin[col].tolist()] for col in rmse_df_coin}
            )
            # Add the coin to the index
            rmse_df_list.index = [coin]
            # Add the data to the dataframe
            rmse_df = pd.concat([rmse_df, rmse_df_list])

        # Save the dataframe to a csv
        rmse_df.to_csv(f"{config.rmse_dir}/{pred}/rmse_{tf}.csv", index=True)

        # Print number on Nan values
        nan_values = rmse_df.isna().sum().sum()
        if nan_values > 0:
            print(f"Number of NaN values in {tf} for {pred}: {nan_values}")


def extended_rmse_df(time_frame: str, avg: bool = False) -> pd.DataFrame:
    # Get RMSE data
    rmse_df = read_rmse_csv(pred=config.extended_pred, time_frame=time_frame, avg=avg)

    # Get the first value of each list in the dataframe -> period 0
    # Change the format that the first column is the period and forget about coin names
    data = {}
    for model in rmse_df.columns:
        # Get the RMSEs for the given model
        data[model] = rmse_df[model].iloc[: config.n_periods].tolist()

    return pd.DataFrame(data, index=range(config.n_periods))


def rmse_heatmap(time_frame: str, pred=config.log_returns_pred):
    if pred == config.extended_pred:
        rmse = extended_rmse_df(time_frame, avg=True)
        decimals = 4
    else:
        rmse = read_rmse_csv(pred, time_frame, avg=True)
        decimals = 2

    plot_rmse_heatmap(
        rmse,
        title=f"RMSE heatmap for {pred} model for {time_frame} time frame",
        round_decimals=decimals,
    )


def plot_rmse_heatmap(
    df: pd.DataFrame,
    title: str,
    flip_colors: bool = False,
    round_decimals: int = 2,
    vmin: float = None,
    vmax: float = None,
    avg_y: bool = True,
    avg_x: bool = True,
):
    if avg_y:
        # Add average column at the right
        df["Average"] = df.mean(axis=1)

    if avg_x:
        df.loc["Average"] = df.mean()

    if flip_colors:
        cmap = "RdYlGn"
    else:
        # Green low, red high
        cmap = "RdYlGn_r"

    # Specify custom range
    if vmin == None:
        vmin = df.min().min()
    if vmax == None:
        vmax = df.max().max()

    plt.figure(figsize=(15, 10))
    plt.rcParams["axes.grid"] = False
    sns.heatmap(
        df,
        annot=True,
        cmap=cmap,
        fmt=f".{round_decimals}f",
        center=0,
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(title)
    plt.show()


def plot_rmse_heatmaps(
    dfs: list[pd.DataFrame],
    title: str,
    titles: list[str],
    flip_colors=False,
    round_decimals=0,
    vmin=None,
    vmax=None,
    avg_y=True,
    avg_x=True,
):
    # Create a subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes = axes.flatten()

    for i, df in enumerate(dfs):
        df = dfs[i]
        ax = axes[i]

        if avg_y:
            df["Average"] = df.mean(axis=1)
        if avg_x:
            df.loc["Average"] = df.mean()

        if flip_colors:
            cmap = "RdYlGn"
        else:
            cmap = "RdYlGn_r"

        if vmin is None:
            vmin = df.min().min()
        if vmax is None:
            vmax = df.max().max()

        sns.heatmap(
            df,
            annot=True,
            cmap=cmap,
            fmt=f".{round_decimals}f",
            center=0,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        ax.grid(False)
        ax.set_title(titles[i])

    plt.tight_layout()
    fig.subplots_adjust(top=0.925)
    fig.suptitle(title)
    plt.show()


def all_models_heatmap(time_frame: str = "1d", preds: list = config.log_preds):
    """
    Plots a heatmap of the RMSE values for all models.

    Parameters
    ----------
    time_frame : str, optional
        The time frame to use for the data, by default "1d"
    log_data : list
        Can be config.log_preds or config.raw_preds
    """

    # Read the RMSE data
    dfs = []
    for pred in preds:
        rmse_df = read_rmse_csv(pred, time_frame, avg=True)
        dfs.append((pred, rmse_df))

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=len(preds), figsize=(20, 10))

    for i, (pred, df) in enumerate(dfs):
        sns.heatmap(df, cmap="coolwarm", ax=axes[i], cbar=False)
        axes[i].set_title(pred)

    # Display a colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sns.heatmap(dfs[0][1], cmap="coolwarm", ax=axes[0], cbar_ax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def forecasting_models_stacked(
    time_frame: str = "1d", preds: list = config.log_preds, coin_on_x: bool = True
):
    """
    Plots a stacked bar plot of the RMSE values for all models.

    Parameters
    ----------
    time_frame : str, optional
        The time frame to use for the data, by default "1d"
    log_data : bool, optional
        If the logarithmic return based models should be used, by default True
    coin_on_x : bool, optional
        If the cryptocurrency coin should be displayed on the x-axis, by default True
    """
    # Read the RMSE data
    dfs = []
    for pred in preds:
        rmse_df = read_rmse_csv(pred, time_frame, avg=True)

        if not coin_on_x:
            rmse_df = rmse_df.T

        dfs.append(rmse_df.T)

    # Aggregate the RMSE values for each cryptocurrency across the models
    df_dict = {x: y.sum(axis=1) for x, y in zip(preds, dfs)}

    # Create a new DataFrame for plotting
    plot_df = pd.DataFrame(df_dict)

    # Plot the Data
    plot_df.plot(kind="bar", stacked=True, figsize=(10, 6))

    if coin_on_x:
        x_label = "Cryptocurrency"
    else:
        x_label = "Model"

    plt.xlabel(x_label)
    plt.ylabel("Aggregated RMSE")
    plt.title("Aggregated RMSE values for each dataset across models")
    plt.legend(title="Datasets")
    plt.tight_layout()
    plt.show()


def get_summed_RMSE(
    time_frame: str = "1d", preds: list = config.log_preds, ignore_models: list = []
) -> pd.DataFrame:
    """
    Helper function for stacked_bar_plot().
    Generates a DataFrame with the summed RMSE values for each model.

    Parameters
    ----------
    time_frame : str, optional
        The time frame to use for this data, by default "1d"
    log_data : bool, optional
        If the logarithmic return based models should be used, by default True
    ignore_models : list, optional
        The models that can be excluded, by default []

    Returns
    -------
    pd.DataFrame
        The summed RMSE values for each model
    """
    # Read the RMSE data
    dfs = []
    for pred in preds:
        rmse_df = read_rmse_csv(pred, time_frame, avg=True)
        rmse_df = rmse_df.sum(axis=0)

        dfs.append(rmse_df)

    # Create dict key = coin, value = list of RMSE values for each model
    df_dict = {}
    for pred in dfs[0].index:
        if pred in ignore_models:
            continue
        df_dict[pred] = [df[pred] for df in dfs]

    # Create a new DataFrame for plotting
    return pd.DataFrame(df_dict, index=preds)


def stacked_bar_plot(
    time_frame: str = "1d", preds: list = config.log_preds, ignore_models: list = []
):
    """
    Plots a stacked bar plot of the RMSE values for all models.

    Parameters
    ----------
    time_frame : str, optional
        The time frame to use for this data, by default "1d"
    log_data : bool, optional
        If the logarithmic return based models should be used, by default True
    ignore_models : list, optional
        The models that can be excluded, by default []
    """
    # Plot the Data
    summed_rmse = get_summed_RMSE(
        time_frame=time_frame, preds=preds, ignore_models=ignore_models
    )

    # Adjust column names
    summed_rmse.rename(columns=config.pred_names, inplace=True)

    summed_rmse.plot(
        kind="bar", stacked=True, figsize=(15, 8), color=plt.cm.Paired.colors
    )

    plt.xlabel("Dataset")
    plt.ylabel("Aggregated RMSE")
    plt.title("Aggregated RMSE values for each model across cryptocurrencies")
    plt.legend(title="Forecasting Models", loc="best", ncols=3)
    plt.tight_layout()
    plt.show()


def stacked_bar_plot_all_tf(preds: list = config.log_preds, ignore_models=[]):
    """
    Plots a stacked bar plot of the RMSE values for all models for all time frames.

    Parameters
    ----------
    log_data : bool, optional
        If the logarithmic return based models should be used, by default True
    ignore_models : list, optional
        The models that can be excluded, by default []
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes = axes.flatten()

    for i, time_frame in enumerate(config.timeframes):
        ax = axes[i]

        plot_df = get_summed_RMSE(
            time_frame=time_frame, preds=preds, ignore_models=ignore_models
        )

        plot_df.plot(kind="bar", stacked=True, ax=ax, color=plt.cm.Paired.colors)

        ax.set_xlabel("Dataset")
        ax.set_ylabel("Aggregated RMSE")
        ax.set_title(f"Aggregated RMSE values for {time_frame} time frame")

        ax.patch.set_edgecolor("black")
        ax.patch.set_linewidth(1)

        # Remove legend for individual subplots
        ax.get_legend().remove()

        # Get handles and labels for the last subplot
        handles, labels = ax.get_legend_handles_labels()

    # Add a single legend for the entire figure
    fig.legend(
        handles,
        labels,
        title="Forecasting Models",
        loc=7,
        ncols=1,
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.88)

    plt.show()


def rmse_comparison(
    time_frame: str = "1d", model_1=config.log_to_raw_pred, model_2=config.raw_pred
):
    """
    Plots a comparison of the RMSE values for two models using a heatmap.

    Parameters
    ----------
    time_frame : str, optional
        The time frame to use for the data, by default "1d"
    model_1 : _type_, optional
        The first model to use for the comparison, by default config.log_to_raw_pred
    model_2 : _type_, optional
        The model to compare model_1 to, by default config.raw_pred
    """
    # Load the data
    rmse_1 = read_rmse_csv(model_1, time_frame, avg=True)
    rmse_2 = read_rmse_csv(model_2, time_frame, avg=True)

    # Calculate the percentual difference
    percentual_difference = ((rmse_2 - rmse_1) / rmse_1) * 100

    # Add average row at the bottom
    percentual_difference.loc["Average"] = percentual_difference.mean()

    # Add average column at the right
    percentual_difference["Average"] = percentual_difference.mean(axis=1)

    # Display or save the resulting table
    print(percentual_difference)

    plot_rmse_heatmap(
        percentual_difference,
        title=f"RMSE percentual comparison between {model_1} model and {model_2} model for {time_frame} time frame",
        flip_colors=True,
    )


def rmse_means(preds: list, time_frame: str = "1d"):
    # Initialize a dictionary to hold the means
    means = {}

    # Read the RMSE data
    dfs = []
    for pred in preds:
        rmse_df = read_rmse_csv(pred, time_frame, avg=True, fill_NaN=True)
        dfs.append(rmse_df)

    for i, df in enumerate(dfs):
        pred = preds[i]  # Assuming the models list and dfs list are aligned
        means[
            pred
        ] = (
            df.mean()
        )  # Calculate the mean for each column and store it in the dictionary

    # Convert to dataframe
    means_df = pd.DataFrame(means)

    # Sort by log_returns
    means_df = means_df.sort_values(by=config.log_returns_pred, ascending=True)

    print(means_df)


def rmse_table(
    pred: list = config.log_returns_pred,
    time_frame: str = config.timeframes[-1],
    coin: str = config.all_coins[0],
    models: list = ["ARIMA", "TCN", "LightGBM"],
):
    """
    Creates a table with rows being the RMSE per period and column the results per forecasting model.

    Parameters
    ----------
    pred : list, optional
        _description_, by default config.log_returns_pred
    time_frame : str, optional
        _description_, by default config.timeframes[-1]
    coin : str, optional
        _description_, by default config.all_coins[0]
    models : list, optional
        _description_, by default ["ARIMA", "TCN", "LightGBM"]
    """

    rmse_df = read_rmse_csv(pred, time_frame, fill_NaN=True)

    # Filter on models
    rmse_df = rmse_df[models]

    # Get the coin that we need the data for
    rmse_df = rmse_df.loc[coin]

    # Restructure dataframe, with period as index and models as columns
    rmse_df = rmse_df.to_frame().T

    def expand_list_in_row(row):
        return pd.DataFrame({col: row[col] for col in rmse_df.columns})

    new_rows = [expand_list_in_row(rmse_df.loc[row]) for row in rmse_df.index]
    new_df = pd.concat(new_rows, keys=rmse_df.index)

    # Reset the index if necessary
    new_df.reset_index(drop=True, inplace=True)

    # Add average row at the end
    new_df.loc["Mean"] = new_df.mean()

    print(f"RMSE results for {coin} on {time_frame} using predictions from: {pred}")
    print(new_df)
