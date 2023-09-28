import pandas as pd

import config
from experiment.rmse import read_rmse_csv, extended_rmse_df, stress_test_rmse_df
from experiment.boxplots import plotly_boxplot, plt_multiple_df_boxplots


def plotly_extended_model_rmse(time_frame):
    df = extended_rmse_df(time_frame=time_frame)

    # Change index
    labels = [f"Until Period {i}" for i in df.index.tolist()]
    df.index = labels

    plotly_boxplot(df=df.T, labels=labels, plot_items=config.ml_models)


def plt_extended_model_rmse(exclude_models=[]):
    dfs = []

    for time_frame in config.timeframes:
        df = extended_rmse_df(time_frame=time_frame)

        # Add ARIMA to the df
        arima_df = read_rmse_csv(pred=config.log_returns_pred, time_frame=time_frame)
        arima_df = pd.DataFrame(
            data={"ARIMA": [arima_df["ARIMA"].str.get(4).to_list()] * config.n_periods},
            index=df.index,
        )
        # Add ARIMA to the df
        df = pd.concat([arima_df, df], axis=1)

        # Drop models in exclude_models
        df = df.drop(exclude_models, axis=1)

        dfs.append(df)

    plt_multiple_df_boxplots(
        dfs,
        outliers_percentile=[80, 90, 95, 99],
        x_label="Training Period Start",
        y_label="RMSE",
        rotate_xlabels=False,
        first_white=True,
    )


def plt_stress_test_rmse():
    dfs = []

    for time_frame in config.timeframes:
        dfs.append(stress_test_rmse_df(time_frame=time_frame))

    plt_multiple_df_boxplots(
        dfs,
        outliers_percentile=[80, 90, 95, 99],
        x_label="Testing Period",
        y_label="RMSE",
        rotate_xlabels=False,
    )
