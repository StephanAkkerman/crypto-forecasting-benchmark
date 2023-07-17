import os
import pandas as pd
from darts.timeseries import TimeSeries
from darts import concatenate
from darts.metrics import rmse
import plotly.graph_objects as go

from hyperopt.search_space import model_config
from hyperopt.config import all_coins, timeframes


def get_predictions(model_name, coin, time_frame):
    preds = []
    tests = []
    rmses = []

    for period in range(5):
        file_path = f"data/models/{model_name}/{coin}/{time_frame}/pred_{period}.csv"
        if not os.path.exists(file_path):
            print(
                f"Warning the following file does not exist: data/models/{model_name}/{coin}/{time_frame}/pred_{period}.csv"
            )
            return None, None, None

        pred = pd.read_csv(file_path)
        pred = TimeSeries.from_dataframe(
            pred, time_col="time", value_cols=["log returns"]
        )

        test = pd.read_csv(
            f"data/models/{model_name}/{coin}/{time_frame}/test_{period}.csv"
        )
        test = TimeSeries.from_dataframe(
            test, time_col="date", value_cols=["log returns"]
        )
        rmses.append(rmse(test, pred))

        # Add it to list
        preds.append(pred)
        tests.append(test)

    preds = concatenate(preds, axis=0)
    tests = concatenate(tests, axis=0)

    return preds, tests, rmses


def all_model_predictions(coin, time_frame):
    model_predictions = {}

    models = list(model_config) + ["ARIMA", "TBATS"]

    for model in models:
        preds, tests, rmses = get_predictions(model, coin, time_frame)

        # If the model does not exist, skip it
        if preds is not None:
            model_predictions[model] = (preds, tests, rmses)

    # Only use the third value in the tuple (the rmse)
    rmses = {model: rmse for model, (_, _, rmse) in model_predictions.items()}
    rmse_df = pd.DataFrame(rmses)

    # Add average row to dataframe
    rmse_df.loc["Average"] = rmse_df.mean()

    return model_predictions, rmse_df


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


def build_rmse_database():
    os.makedirs("data/analysis", exist_ok=True)

    for tf in timeframes:
        rmse_df = pd.DataFrame()
        for coin in all_coins:
            # Get the predictions
            _, rmse_df_coin = all_model_predictions(coin, tf)
            rmse_df_list = pd.DataFrame(
                {col: [rmse_df_coin[col].tolist()] for col in rmse_df_coin}
            )
            # print(rmse_df_list)
            # Add the coin to the index
            rmse_df_list.index = [coin]
            # Add the data to the dataframe
            rmse_df = pd.concat([rmse_df, rmse_df_list])

        # Save the dataframe to a csv
        rmse_df.to_csv(f"data/analysis/rmse_{tf}.csv", index=True)

        # Print number on Nan values
        print(f"Number of NaN values in {tf}: {rmse_df.isna().sum().sum()}")


def models_boxplot():
    pass
