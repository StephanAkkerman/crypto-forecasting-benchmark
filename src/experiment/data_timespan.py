import pandas as pd

import config
from experiment.rmse import read_rmse_csv, extended_rmse_df
from experiment.boxplots import plotly_boxplot, plt_multiple_df_boxplots


def plotly_extended_model_rmse(time_frame):
    df = extended_rmse_df(time_frame=time_frame)

    # Change index
    labels = [f"Until Period {i}" for i in df.index.tolist()]
    df.index = labels

    plotly_boxplot(df=df.T, labels=labels, plot_items=config.ml_models)


def plt_extended_model_rmse(exclude_models=["NBEATS"]):
    dfs = []

    for time_frame in config.timeframes:
        # Get RMSE data
        rmse_df = read_rmse_csv(pred=config.extended_pred, time_frame=time_frame)

        data = {}
        for model in rmse_df.columns:
            if model in exclude_models:
                continue

            # Get the RMSEs for the given model
            data[model] = rmse_df[model].iloc[: config.n_periods].tolist()

        df = pd.DataFrame(data)

        # Rename indices
        df.index = [f"Period {i+1}" for i in range(config.n_periods)]
        dfs.append(df)

    plt_multiple_df_boxplots(
        dfs,
        outliers_percentile=100,
        x_label="Period",
        y_label="Logarithmic Returns",
        rotate_xlabels=False,
    )
