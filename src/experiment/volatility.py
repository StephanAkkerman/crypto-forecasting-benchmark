import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import config
from data_analysis.volatility import (
    get_tf_percentile,
    get_volatility,
)
from experiment.train_test import get_train_test
from experiment.rmse import read_rmse_csv


def strip_quotes(cell):
    return [i.strip("'") for i in cell]


def read_volatility_csv(model: str, time_frame: str):
    df = pd.read_csv(
        f"{config.volatility_dir}/{model}/vol_{time_frame}.csv", index_col=0
    )

    # Convert string to list
    df = df.applymap(lambda x: x.strip("[]").split(", "))

    # Remove ' from string for class columns
    df[["train_volatility_class", "test_volatility_class"]] = df[
        ["train_volatility_class", "test_volatility_class"]
    ].applymap(lambda x: [i.strip("'") for i in x])

    # Convert list of strings to list of floats
    df[["train_volatility", "test_volatility"]] = df[
        ["train_volatility", "test_volatility"]
    ].applymap(lambda x: [float(i) for i in x])

    return df


def get_volatility_class(volatility, percentile25, percentile75):
    if volatility < percentile25:
        return "low"
    elif volatility > percentile75:
        return "high"
    else:
        return "normal"


def create_volatility_data(model: str = config.log_returns_model):
    """Analyze and plot the impact of volatility on RMSE for different train-test splits."""
    save_loc = f"{config.volatility_dir}/{model}"
    os.makedirs(save_loc, exist_ok=True)

    for time_frame in config.timeframes:
        # Calculate the percentiles for this time frame
        percentile25, percentile75 = get_tf_percentile(time_frame=time_frame)

        volatility_df = pd.DataFrame()

        for coin in config.all_coins:
            # Get the volatility for the coin
            volatility = get_volatility(coin=coin, time_frame=time_frame)

            # Get the train and test times
            trains, tests, _ = get_train_test(coin=coin, time_frame=time_frame)

            # Save the data here
            train_volatilty_class = []
            test_volatilty_class = []
            train_volatilty = []
            test_volatilty = []

            # Loop over each period
            for train, test in zip(trains, tests):
                # Determine the train and test volatility
                vol_train = volatility.loc[train.start_time() : train.end_time()]
                vol_test = volatility.loc[test.start_time() : test.end_time()]

                # Calculate the mean volatility for train and test
                mean_vol_train = vol_train.mean().values[0]
                mean_vol_test = vol_test.mean().values[0]

                # Calculate the volatility class for train and test
                train_volatilty_class.append(
                    get_volatility_class(
                        volatility=mean_vol_train,
                        percentile25=percentile25,
                        percentile75=percentile75,
                    )
                )

                test_volatilty_class.append(
                    get_volatility_class(
                        volatility=mean_vol_test,
                        percentile25=percentile25,
                        percentile75=percentile75,
                    )
                )

                # Also add 2 columns without classification and just the mean number
                train_volatilty.append(mean_vol_train)
                test_volatilty.append(mean_vol_test)

            # Create a dataframe for the coin
            coin_df = pd.DataFrame(
                data=[
                    {
                        "train_volatility_class": train_volatilty_class,
                        "test_volatility_class": test_volatilty_class,
                        "train_volatility": train_volatilty,
                        "test_volatility": test_volatilty,
                    }
                ],
                index=[coin],
            )

            # Add to df
            volatility_df = pd.concat([volatility_df, coin_df])

        # Save: volatility classification
        volatility_df.to_csv(f"{save_loc}/vol_{time_frame}.csv")


def create_all_volatility_data():
    for model in [
        config.log_returns_model,
        config.log_to_raw_model,
        config.extended_model,
        config.extended_to_raw_model,
        config.raw_model,
        config.raw_to_log_model,
        config.scaled_model,
        config.scaled_to_log_model,
        config.scaled_to_raw_model,
        config.scaled_to_raw_model,
    ]:
        print(f"Creating volatility data for {model}")
        create_volatility_data(model=model)


def boxplot():
    pass


def model_boxplot(model: str = config.log_returns_model, time_frame: str = "1d"):
    """
    Shows the correlation between volatility and RMSE for each forecasting model.

    Parameters
    ----------
    model : str, optional
        The model output to use, by default config.log_returns_model
    time_frame : str, optional
        The time frame to use, by default "1d"
    """

    # Read the data
    rmse_df = read_rmse_csv(model, time_frame=time_frame)
    vol_df = read_volatility_csv(model, time_frame=time_frame)

    list_of_dfs = []

    # Loop through each model to populate all_flattened_df
    for model_name in rmse_df.columns:  # Loop through all models
        rmse = rmse_df[model_name]

        # Create a temporary DataFrame for this model
        temp_vol_df = vol_df.copy()

        temp_vol_df["rmse"] = rmse
        temp_vol_df["coin"] = temp_vol_df.index
        temp_vol_df["model"] = model_name  # Add the model name

        # Reset index and flatten the DataFrame
        temp_vol_df.reset_index(inplace=True, drop=True)
        flattened_df = temp_vol_df.apply(lambda x: x.explode())

        list_of_dfs.append(flattened_df)

    # Create a grid of subplots
    fig, axes = plt.subplots(2, 6, figsize=(22, 10))
    axes = axes.flatten()

    # Loop through the list of DataFrames and axes to create a boxplot for each
    for i, (df, ax) in enumerate(zip(list_of_dfs, axes)):
        sns.boxplot(
            x="train_volatility_class",
            y="rmse",
            hue="test_volatility_class",
            data=df,
            palette="Set3",
            ax=ax,
            order=["low", "normal", "high"],
        )
        # ax.set_xlabel('Volatility Class During Training')
        # ax.set_ylabel('RMSE')
        ax.set_title(f"Model: {df['model'].iloc[0]}")
        ax.get_legend().remove()

    # Remove any remaining empty subplots
    for ax in axes[len(list_of_dfs) :]:
        ax.remove()

    # Add a global legend
    (
        handles,
        labels,
    ) = (
        ax.get_legend_handles_labels()
    )  # We can use the handles and labels of the last subplot
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.99999),
        title="Test Volatility",
        ncols=3,
    )

    # Add a super title for the entire figure
    plt.suptitle(
        "Boxplots of RMSE by Volatility Class During Training and Testing for Multiple DataFrames",
    )
    plt.tight_layout()
    plt.show()


def coin_boxplot(
    model: str = config.log_returns_model,
    time_frame: str = "1d",
    forecasting_model="ARIMA",
):
    # Read the data
    rmse_df = read_rmse_csv(model, time_frame=time_frame)
    vol_df = read_volatility_csv(model, time_frame=time_frame)

    # Add rmse to vol_df
    vol_df["rmse"] = rmse_df[forecasting_model]

    # Explode the lists into individual rows
    df_exploded = (
        vol_df.apply(pd.Series.explode).reset_index().rename(columns={"index": "Coin"})
    )

    # Create the multi-level boxplot using Seaborn
    g = sns.catplot(
        x="train_volatility_class",
        y="rmse",
        hue="test_volatility_class",
        col="Coin",
        kind="box",
        data=df_exploded,
        palette="Set3",
        height=6,
        aspect=1,
        col_wrap=6,
    )

    # Add labels and title
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(
        "Multi-Level Boxplot of RMSE by Volatility Class During Training and Testing, Grouped by Coin"
    )

    # Show the plot
    plt.show()


def volatility_rmse_heatmap(
    model: str = config.log_returns_model, time_frame: str = "15m"
):
    """
    Plots the mean RMSE for each combination of train and test volatility class.

    Parameters
    ----------
    model : str, optional
        The name of the model output to use, by default config.log_returns_model
    time_frame : str, optional
        The time frame to use, by default "1d"
    """

    # Read the data
    rmse_df = read_rmse_csv(
        model, time_frame=time_frame
    )  # Assuming this reads RMSE for all models
    vol_df = read_volatility_csv(model, time_frame=time_frame)

    # Initialize an empty DataFrame to store flattened data
    all_flattened_df = pd.DataFrame()

    # Loop through each model to populate all_flattened_df
    for model_name in rmse_df.columns:  # Loop through all models
        rmse = rmse_df[model_name]

        # Create a temporary DataFrame for this model
        temp_vol_df = vol_df.copy()

        temp_vol_df["rmse"] = rmse
        temp_vol_df["coin"] = temp_vol_df.index
        temp_vol_df["model"] = model_name  # Add the model name

        # Reset index and flatten the DataFrame
        temp_vol_df.reset_index(inplace=True, drop=True)
        flattened_df = temp_vol_df.apply(lambda x: x.explode())

        # Append to all_flattened_df
        all_flattened_df = pd.concat([all_flattened_df, flattened_df])

    # You can aggregate by mean, or other functions like 'median', 'sum', etc.
    pivot_table = pd.pivot_table(
        all_flattened_df,
        values="rmse",
        index=["train_volatility_class", "model"],  # Include model in index
        columns=["test_volatility_class"],
        aggfunc="mean",
    )

    # Reordering index and columns
    pivot_table = pivot_table.reorder_levels(
        ["train_volatility_class", "model"]
    ).sort_index()

    # Specifying the desired order for columns and index
    desired_order = ["low", "normal", "high"]

    # Reordering columns
    pivot_table = pivot_table[desired_order]

    # Reordering the index level 'train_volatility_class'
    pivot_table = pivot_table.reorder_levels(["train_volatility_class", "model"]).loc[
        desired_order
    ]

    # Create the heatmap
    _, ax1 = plt.subplots(figsize=(16, 10))
    ax1.grid(False)
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", ax=ax1)
    plt.rcParams["axes.grid"] = False
    plt.title("Impact of Train and Test Volatility on RMSE Across Models")
    ax1.set_xlabel("Volatility Class")

    # Invert the y-axis
    ax1.invert_yaxis()

    # Create a twin y-axis
    ax2 = ax1.twinx()

    # Get the current y-tick labels from the first axis
    current_labels = [item.get_text() for item in ax1.get_yticklabels()]

    # Create new labels for the second axis that contain only the model names
    ax1_labels = [label.split("-")[1] for label in current_labels]
    ax2_labels = [label.split("-")[0].capitalize() for label in current_labels]

    ax1.set_yticklabels(ax1_labels)

    ax1_ylim = ax1.get_ylim()
    ax2.set_ylim(ax1_ylim)

    # Now set the ticks
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_yticklabels(ax2_labels)

    # Capitalize x-axis labels
    ax1.set_xticklabels(
        [label.get_text().capitalize() for label in ax1.get_xticklabels()]
    )

    # Set y-axis labels
    ax1.set_ylabel("Forecasting Model")

    plt.show()


# Other options:
# multi-level boxplot
# 3d scatter plot
# facet grid scatter plot
