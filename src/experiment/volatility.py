import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from data_analysis.volatility_analysis import get_tf_percentile, get_volatility
from experiment.rmse import assign_mcap_category, read_rmse_csv
from experiment.train_test import get_train_test


def strip_quotes(cell):
    return [i.strip("'") for i in cell]


def read_volatility_csv(time_frame: str, add_mcap: bool = False):
    file_loc = f"{config.volatility_dir}/vol_{time_frame}.csv"

    # Check if the file exists
    if not os.path.exists(file_loc):
        print(f"Did not found volatility data for {file_loc}. Creating it...")
        create_volatility_data()

    df = pd.read_csv(file_loc, index_col=0)

    # Convert string to list
    df = df.applymap(lambda x: x.strip("[]").split(", "))

    # Remove ' from string for class columns
    df[
        ["train_volatility_class", "test_volatility_class", "period_volatility_class"]
    ] = df[
        ["train_volatility_class", "test_volatility_class", "period_volatility_class"]
    ].applymap(
        lambda x: [i.strip("'") for i in x]
    )

    # Convert list of strings to list of floats
    df[["train_volatility", "test_volatility", "period_volatility"]] = df[
        ["train_volatility", "test_volatility", "period_volatility"]
    ].applymap(lambda x: [float(i) for i in x])

    if add_mcap:
        df["mcap category"] = df.index.map(assign_mcap_category)

    return df


def get_volatility_class(volatility, percentile25, percentile75):
    if volatility < percentile25:
        return "low"
    elif volatility > percentile75:
        return "high"
    else:
        return "normal"


def create_volatility_data():
    os.makedirs(config.volatility_dir, exist_ok=True)

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
            period_volatility_class = []

            train_volatilty = []
            test_volatilty = []
            period_volatility = []

            # Loop over each period
            for train, test in zip(trains, tests):
                # Determine the train and test volatility
                vol_train = volatility.loc[train.start_time() : train.end_time()]
                vol_test = volatility.loc[test.start_time() : test.end_time()]
                vol_period = volatility.loc[train.start_time() : test.end_time()]

                # Calculate the mean volatility for train and test
                mean_vol_train = vol_train.mean().values[0]
                mean_vol_test = vol_test.mean().values[0]
                mean_vol_period = vol_period.mean().values[0]

                # Calculate the volatility class for train, test, and period
                for vol_list, vol in [
                    (train_volatilty_class, mean_vol_train),
                    (test_volatilty_class, mean_vol_test),
                    (period_volatility_class, mean_vol_period),
                ]:
                    vol_list.append(
                        get_volatility_class(
                            volatility=vol,
                            percentile25=percentile25,
                            percentile75=percentile75,
                        )
                    )

                # Also add 2 columns without classification and just the mean number
                train_volatilty.append(mean_vol_train)
                test_volatilty.append(mean_vol_test)
                period_volatility.append(mean_vol_period)

            # Create a dataframe for the coin
            coin_df = pd.DataFrame(
                data=[
                    {
                        "train_volatility_class": train_volatilty_class,
                        "test_volatility_class": test_volatilty_class,
                        "period_volatility_class": period_volatility_class,
                        "train_volatility": train_volatilty,
                        "test_volatility": test_volatilty,
                        "period_volatility": period_volatility,
                    }
                ],
                index=[coin],
            )

            # Add to df
            volatility_df = pd.concat([volatility_df, coin_df])

        # Save: volatility classification
        volatility_df.to_csv(f"{config.volatility_dir}/vol_{time_frame}.csv")


def boxplot(
    pred: str = config.log_returns_pred,
    time_frame: str = "1d",
    log_scale: bool = False,
    ignore_outliers: bool = True,
):
    """
    Shows the correlation between volatility and RMSE for each forecasting model, aggregated into one boxplot.

    Parameters
    ----------
    pred : str, optional
        The prediction output to use, by default config.log_returns_pred
    time_frame : str, optional
        The time frame to use, by default "1d"
    """

    # Read the data
    rmse_df = read_rmse_csv(pred, time_frame=time_frame)
    vol_df = read_volatility_csv(time_frame=time_frame)

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

    # Combine all DataFrames in list_of_dfs into one DataFrame
    combined_df = pd.concat(list_of_dfs, ignore_index=True)

    # Create the single boxplot
    plt.figure(figsize=(16, 10))
    ax = sns.boxplot(
        x="train_volatility_class",
        y="rmse",
        hue="test_volatility_class",
        data=combined_df,
        palette="Set3",
        order=["low", "normal", "high"],
    )

    if log_scale:
        ax.set_yscale("log")

    if ignore_outliers:
        # Calculate the 5th and 95th percentiles for the y-axis limits
        ymax = np.percentile(combined_df["rmse"].dropna(), 95)

        # Set the y-axis limits
        ax.set_ylim(combined_df["rmse"].min(), ymax)

    plt.xlabel("Volatility Class During Training")
    plt.ylabel("RMSE")
    plt.title(
        "Boxplots of RMSE by Volatility Class During Training and Testing (Aggregated)"
    )
    plt.legend(title="Test Volatility", loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


def model_boxplot(pred: str = config.log_returns_pred, time_frame: str = "1d"):
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
    rmse_df = read_rmse_csv(pred, time_frame=time_frame)
    vol_df = read_volatility_csv(time_frame=time_frame)

    list_of_dfs = []

    # Loop through each model to populate all_flattened_df
    for forecasting_model in rmse_df.columns:  # Loop through all models
        rmse = rmse_df[forecasting_model]

        # Create a temporary DataFrame for this model
        temp_vol_df = vol_df.copy()

        temp_vol_df["rmse"] = rmse
        temp_vol_df["coin"] = temp_vol_df.index
        temp_vol_df["model"] = forecasting_model  # Add the model name

        # Reset index and flatten the DataFrame
        temp_vol_df.reset_index(inplace=True, drop=True)
        flattened_df = temp_vol_df.apply(lambda x: x.explode())

        list_of_dfs.append(flattened_df)

    # Create a grid of subplots
    fig, axes = plt.subplots(2, 6, figsize=(22, 10))
    axes = axes.flatten()

    # Loop through the list of DataFrames and axes to create a boxplot for each
    for _, (df, ax) in enumerate(zip(list_of_dfs, axes)):
        sns.boxplot(
            x="train_volatility_class",
            y="rmse",
            hue="test_volatility_class",
            data=df,
            palette="Set3",
            ax=ax,
            order=["low", "normal", "high"],
        )
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
        f"Boxplots of RMSE by Volatility Class During Training and Testing for Multiple DataFrames. Time Frame: {time_frame}",
    )
    plt.tight_layout()
    plt.show()


def coin_boxplot(
    pred: str = config.log_returns_pred,
    time_frame: str = "1d",
    forecasting_model="ARIMA",
):
    # Read the data
    rmse_df = read_rmse_csv(pred, time_frame=time_frame)
    vol_df = read_volatility_csv(time_frame=time_frame)

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
        order=["low", "normal", "high"],
    )

    # Add labels and title
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(
        f"Boxplot of RMSE by Volatility Class During Training and Testing, Grouped by Coin. Forecasting Model: {forecasting_model}"
    )

    # Show the plot
    plt.show()


def volatility_rmse_heatmap(
    pred: str = config.log_returns_pred, exclude_model=["NBEATS", "NHiTS", "TFT"]
):
    """
    Plots the mean RMSE for each combination of train and test volatility class.

    Parameters
    ----------
    pred : str, optional
        The prediction output to use, by default config.log_returns_pred
    time_frame : str, optional
        The time frame to use, by default "1d"
    """
    # plt.style.use("dark_background")

    _, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()  # Flatten the 2D array to 1D for easy iteration

    for idx, time_frame in enumerate(config.timeframes):
        # Read the data
        rmse_df = read_rmse_csv(
            pred, time_frame=time_frame
        )  # Assuming this reads RMSE for all models
        vol_df = read_volatility_csv(time_frame=time_frame)

        # Initialize an empty DataFrame to store flattened data
        all_flattened_df = pd.DataFrame()

        # Loop through each model to populate all_flattened_df
        for model_name in rmse_df.columns:  # Loop through all models
            if model_name in exclude_model:
                continue

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
        pivot_table = pivot_table.reorder_levels(
            ["train_volatility_class", "model"]
        ).loc[desired_order]

        # Create the heatmap
        ax1 = axes[idx]
        ax1.set_title(config.tf_names[idx])
        ax1.grid(False)

        sns.heatmap(
            pivot_table,
            annot=True,
            cmap="YlGnBu",
            ax=ax1,
            cbar_kws={"pad": 0.1},
            vmax=0.0018 if time_frame == "1m" else None,
        )
        plt.rcParams["axes.grid"] = False

        ax1.set_xlabel("Test Volatility Class")

        # Invert the y-axis
        ax1.invert_yaxis()

        # Create a twin y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Train Volatility Class")

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

    # plt.title(
    #    "Impact of Train and Test Volatility on RMSE Across Models"
    # )
    plt.tight_layout()
    plt.show()


def mcap_rmse_boxplot(
    pred: str = config.log_returns_pred,
    ignore_model=[],
    log_scale: bool = False,
    remove_outliers: bool = True,
    fontsize: int = 12,
    dark_mode: bool = True,
):
    if dark_mode:
        plt.style.use("dark_background")
        colors = plt.cm.Dark2.colors
        PROPS = {
            "boxprops": {"edgecolor": "white"},
            "medianprops": {"color": "white"},
            "whiskerprops": {"color": "white"},
            "capprops": {"color": "white"},
        }
    else:
        colors = plt.cm.Accent.colors
        PROPS = {}

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    for i, time_frame in enumerate(config.timeframes):
        df = read_rmse_csv(
            pred,
            time_frame=time_frame,
            avg=True,
            add_mcap=True,
            ignore_model=ignore_model,
        )

        melted_df = pd.melt(df, id_vars="mcap category")

        sns.boxplot(
            x="mcap category",
            y="value",
            hue="variable",
            data=melted_df,
            palette=colors,
            ax=axes[i],
            order=["Small", "Mid", "Large"],
            **PROPS,
        )

        axes[i].set_title(config.tf_names[i])
        axes[i].set_xlabel("Market Cap Category")
        axes[i].set_ylabel("RMSE")

        # Remove the legend from individual subplots
        axes[i].get_legend().remove()

        if log_scale:
            axes[i].set_yscale("log")

        if remove_outliers:
            Q1 = melted_df["value"].quantile(0.25)
            Q3 = melted_df["value"].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if lower_bound < 0:
                lower_bound = 0
            axes[i].set_ylim(lower_bound, upper_bound)

    # Adjust the layout to leave space for an upper center legend
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Add a single global legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),  # adjust the numbers as per your requirement
        title="Forecasting Model",
        ncols=len(handles),
        fontsize=fontsize,
        title_fontsize=fontsize,
    )

    plt.show()


def mcap_vol_boxplot(dark_mode: bool = True):
    """
    Creates a boxplot of volatility for each market cap category.
    Creates this for each time frame.
    """
    if dark_mode:
        plt.style.use("dark_background")
        colors = plt.cm.Dark2.colors
        PROPS = {
            "boxprops": {"edgecolor": "white"},
            "medianprops": {"color": "white"},
            "whiskerprops": {"color": "white"},
            "capprops": {"color": "white"},
        }
    else:
        colors = plt.cm.Accent.colors
        PROPS = {}

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    for i, time_frame in enumerate(config.timeframes):
        df = read_volatility_csv(time_frame=time_frame)

        flat_data = []
        for index, row in df.iterrows():
            flat_data.extend([(index, val) for val in row["period_volatility"]])

        df_flat = pd.DataFrame(flat_data, columns=["Coin", "period_volatility"])
        df_flat["Market Cap Category"] = df_flat["Coin"].map(assign_mcap_category)

        sns.boxplot(
            x="Market Cap Category",
            y="period_volatility",
            data=df_flat,
            palette=colors[:3],
            ax=axes[i],
            order=["Small", "Mid", "Large"],
            **PROPS,
        )

        axes[i].set_title(config.tf_names[i])
        axes[i].set_xlabel("Market Cap Category")
        axes[i].set_ylabel("Volatility")

    plt.tight_layout()
    plt.show()


def mcap_rmse_heatmap(pred: str = config.log_returns_pred, ignore_model=[]):
    _, axes = plt.subplots(2, 2, figsize=(20, 10))  # Create a 2x2 grid of subplots
    axes = axes.flatten()  # Flatten the 2x2 grid to a 1D array

    all_values = (
        []
    )  # List to collect all RMSE values across timeframes and mcap categories
    for i, time_frame in enumerate(config.timeframes):
        df = read_rmse_csv(pred, time_frame=time_frame, avg=True, add_mcap=True)

        # Grouping by 'mcap category' and calculating the mean RMSE for each group
        grouped_df = df.groupby("mcap category").mean()

        # Collect all RMSE values
        all_values.extend(grouped_df.values.flatten())

    for i, time_frame in enumerate(config.timeframes):
        df = read_rmse_csv(pred, time_frame=time_frame, avg=True, add_mcap=True)

        # Grouping by 'mcap category' and calculating the mean RMSE for each group
        grouped_df = df.groupby("mcap category").mean()

        # Calculate the maximum value for the color bar
        vmax = [item for sublist in grouped_df.values for item in sublist]
        vmax = np.percentile(vmax, 75)

        sns.heatmap(
            grouped_df,
            annot=True,
            cmap="YlGnBu",
            ax=axes[i],  # Specify which subplot to use
            # vmin=vmin,  # Minimum value for color bar
            vmax=vmax,
        )
        axes[i].grid(False)

        axes[i].set_title(f"RMSE Heatmap for {pred} - {time_frame}")
        axes[i].set_xlabel("Forecasting Model")
        axes[i].set_ylabel("Market Cap Category")

    plt.tight_layout()
    plt.show()


def mcap_volatility_heatmap():
    fig, axes = plt.subplots(2, len(config.timeframes), figsize=(20, 10))

    # Create an axes object for the colorbar
    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])

    max_value = 0
    for i, time_frame in enumerate(config.timeframes):
        train_cbar = False
        test_cbar = False

        df = read_volatility_csv(time_frame=time_frame, add_mcap=True)

        # Initialize empty lists to hold the flattened records
        flattened_data = {
            "coin": [],
            "mcap_category": [],
            "volatility_class": [],
            "set_type": [],
        }

        # Flatten the DataFrame
        for idx, row in df.iterrows():
            mcap_category = row["mcap category"]
            train_volatility_classes = row["train_volatility_class"]
            test_volatility_classes = row["test_volatility_class"]

            for train_class in train_volatility_classes:
                flattened_data["coin"].append(idx)
                flattened_data["mcap_category"].append(mcap_category)
                flattened_data["volatility_class"].append(train_class)
                flattened_data["set_type"].append("Train")

            for test_class in test_volatility_classes:
                flattened_data["coin"].append(idx)
                flattened_data["mcap_category"].append(mcap_category)
                flattened_data["volatility_class"].append(test_class)
                flattened_data["set_type"].append("Test")

        # Create a new DataFrame from the flattened data
        df_flattened = pd.DataFrame(flattened_data)

        # Create crosstab tables
        train_crosstab = pd.crosstab(
            df_flattened[df_flattened["set_type"] == "Train"]["mcap_category"],
            df_flattened[df_flattened["set_type"] == "Train"]["volatility_class"],
        ).reindex(columns=["low", "normal", "high"])

        test_crosstab = pd.crosstab(
            df_flattened[df_flattened["set_type"] == "Test"]["mcap_category"],
            df_flattened[df_flattened["set_type"] == "Test"]["volatility_class"],
        ).reindex(columns=["low", "normal", "high"])

        train_max = train_crosstab.max().max()
        test_max = test_crosstab.max().max()

        if train_max > max_value:
            max_value = train_max
            train_cbar = True
        if test_max > max_value:
            max_value = test_max
            test_cbar = True

        # Calculate subplot row and column indices
        row_idx = i // 2
        col_idx = i % 2  # Adjusted to fit within the 2x4 grid

        # Plotting heatmaps for Train set
        ax1 = axes[row_idx, col_idx * 2]
        sns.heatmap(
            train_crosstab,
            annot=True,
            cmap="coolwarm",
            fmt=".0f",
            ax=ax1,
            cbar=train_cbar,
            cbar_ax=cbar_ax,
        ).grid(False)
        ax1.set_title(f"{time_frame} - Train")
        ax1.set_xlabel("Volatility Class")
        ax1.set_ylabel("Market Cap Category")

        # Plotting heatmaps for Test set
        ax2 = axes[row_idx, col_idx * 2 + 1]
        sns.heatmap(
            test_crosstab,
            annot=True,
            cmap="coolwarm",
            fmt=".0f",
            ax=ax2,
            cbar=test_cbar,
            cbar_ax=cbar_ax,
        ).grid(False)
        ax2.set_title(f"{time_frame} - Test")
        ax2.set_xlabel("Volatility Class")
        ax2.set_ylabel("Market Cap Category")

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for the colorbar

    # Add title
    fig.subplots_adjust(top=0.9)
    fig.suptitle(f"Heatmap of Market Cap vs Volatility Class for Train and Test Sets")

    plt.show()


def get_mean_vol(coin, time_frame, percentile25, percentile75) -> float:
    volatility = get_volatility(coin=coin, time_frame=time_frame)

    vol_classes = []

    for vol in volatility.values:
        vol_class = get_volatility_class(
            volatility=vol,
            percentile25=percentile25,
            percentile75=percentile75,
        )
        vol_classes.append(vol_class)

    return vol_classes


def tf_mean_vol(time_frame: str) -> pd.DataFrame:
    volatilities = {}
    percentile25, percentile75 = get_tf_percentile(time_frame=time_frame)

    for coin in config.all_coins:
        volatilities[coin] = get_mean_vol(coin, time_frame, percentile25, percentile75)

    return pd.DataFrame([volatilities])


def tf_significance():
    dfs = []
    for tf in config.timeframes:
        df = tf_mean_vol(tf)
        # Add timeframe column
        df["Time Frame"] = tf
        dfs.append(df)

    df = pd.concat(dfs)

    # Calculate the frequency of each category for each time frame
    results = {}

    # Iterate over each cryptocurrency
    for coin in config.all_coins:
        results[coin] = {}

        # Calculate the frequency of each category for each time frame
        for index, row in df.iterrows():
            time_frame = row["Time Frame"]
            categories = row[coin]
            counts = Counter(categories)
            total_counts = sum(counts.values())

            # Calculate percentage for each category
            percentages = {k: (v / total_counts) * 100 for k, v in counts.items()}

            results[coin][time_frame] = percentages

    # Convert the results to a DataFrame for better display
    result_df = pd.DataFrame.from_dict(
        {
            (coin, tf): values
            for coin, time_frames in results.items()
            for tf, values in time_frames.items()
        },
        orient="index",
    ).fillna(0)
    print(result_df)
