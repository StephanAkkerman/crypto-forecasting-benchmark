import os
import pandas as pd

# Models
from darts.models import (
    StatsForecastAutoARIMA,
    RNNModel,
    TCNModel,
    NBEATSModel,
    TFTModel,
    RandomForest,
    XGBModel,
    LightGBMModel,
    NHiTSModel,
    TBATS,
    Prophet,
)

# Local imports
from experiment.train_test import get_train_test
from hyperopt.analysis import best_hyperparameters
from hyperopt.config import all_coins, timeframes
from hyperopt.search_space import model_config


def get_model(model_name, coin, time_frame):
    # TODO: Also add the model unspecific parameters

    if model_name == "ARIMA":
        return StatsForecastAutoARIMA(
            start_p=0,
            start_q=0,
            start_P=0,
            start_Q=0,
            max_p=5,
            max_d=5,
            max_q=5,
            max_P=5,
            max_Q=5,
        )
    elif model_name == "RandomForest":
        return RandomForest(**best_hyperparameters(model_name, coin, time_frame))
    elif model_name == "XGB":
        return XGBModel(**best_hyperparameters(model_name, coin, time_frame))
    elif model_name == "LightGBM":
        return LightGBMModel(**best_hyperparameters(model_name, coin, time_frame))
    elif model_name == "Prophet":
        return Prophet(**best_hyperparameters(model_name, coin, time_frame))
    elif model_name == "TBATS":
        return TBATS(use_arma_errors=None)
    elif model_name == "NBEATS":
        return NBEATSModel(
            **best_hyperparameters(model_name, coin, time_frame), model_name=model_name
        )
    elif model_name == "RNN":
        return RNNModel(
            **best_hyperparameters(model_name, coin, time_frame), model_name=model_name
        )
    elif model_name == "LSTM":
        return RNNModel(
            **best_hyperparameters(model_name, coin, time_frame),
            model_name=model_name,
            model="LSTM",
        )
    elif model_name == "GRU":
        return RNNModel(
            **best_hyperparameters(model_name, coin, time_frame),
            model_name=model_name,
            model="GRU",
        )
    elif model_name == "TCN":
        return TCNModel(
            **best_hyperparameters(model_name, coin, time_frame), model_name=model_name
        )
    elif model_name == "TFT":
        return TFTModel(
            **best_hyperparameters(model_name, coin, time_frame), model_name=model_name
        )
    elif model_name == "NHiTS":
        return NHiTSModel(
            **best_hyperparameters(model_name, coin, time_frame), model_name=model_name
        )
    else:
        raise ValueError(f"Model {model_name} is not supported.")


def generate_forecasts(
    model_name: str, coin: str, time_frame: str, n_periods=5, show_plot=True
):
    # Get the training and testing data for each period
    train_set, test_set, time_series = get_train_test(
        coin=coin,
        time_frame=time_frame,
        n_periods=n_periods,
    )

    model = get_model(model_name, coin, time_frame)

    # Certain models need to be retrained for each period
    retrain = False
    train_length = None
    if model_name in ["Prophet", "TBATS", "ARIMA"]:
        retrain = True
        train_length = len(train_set[period])

    for period in range(n_periods):
        model.fit(series=time_series[period])

        pred = model.historical_forecasts(
            time_series[period],
            start=len(train_set[period]),
            forecast_horizon=1,  # 1 step ahead forecasting
            stride=1,  # 1 step ahead forecasting
            retrain=retrain,
            train_length=train_length,
            verbose=False,
        )

        # Save all important information
        pred.pd_dataframe().to_csv(
            f"data/models/{model_name}/{coin}/{time_frame}/pred_{period}.csv"
        )
        train_set[period].pd_dataframe().to_csv(
            f"data/models/{model_name}/{coin}/{time_frame}/train_{period}.csv"
        )
        test_set[period].pd_dataframe().to_csv(
            f"data/models/{model_name}/{coin}/{time_frame}/test_{period}.csv"
        )

        # Save ARIMA parameters
        if model_name == "ARIMA":
            params_df = pd.DataFrame(
                model.model.model_["arma"], columns=["p", "d", "q", "P", "D", "Q", "C"]
            )
            params_df.to_csv(
                f"data/models/{model_name}/{coin}/{time_frame}/params_{period}.csv",
                index=False,
            )

        # Print information
        print(
            f"Saved {model_name} forecast in data/models/{model_name}/{coin}/{time_frame}/ for period {period}"
        )


def forecast_model(model_name):
    for coin in all_coins:
        for time_frame in timeframes:
            # Create directories
            os.makedirs(f"data/models/{model_name}/{coin}/{time_frame}", exist_ok=True)

            generate_forecasts(
                model_name, coin, time_frame, n_periods=5, show_plot=False
            )


def forecast_all():
    models = list(model_config) + ["ARIMA", "TBATS"]

    for model in models:
        forecast_model(model)


def test_models():
    for model in list(model_config) + ["ARIMA", "TBATS"]:
        for coin in all_coins:
            for time_frame in timeframes:
                print(f"Testing {model} for {coin} {time_frame}")
                get_model(model, coin, time_frame)
